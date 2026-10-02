#!/usr/bin/env python3
"""Collect every Maritime3D benchmark experiment into one results document.

    python3 tools/collect_bench_results.py            # writes the files below
    python3 tools/collect_bench_results.py --stdout   # also print the markdown

Writes projects/Maritime3D/RESULTS_BENCH.md, results_bench.csv and
results_bench.json. Sources:
* val / test: work_dirs/eval/<name>/<split>/ (tools/test.py with every
  MaritimeMetric breakdown, final checkpoint of each run; a single
  evaluation, so its summary line is exact);
* latency / memory: the same evaluation logs (8 A800, 1 frame per GPU,
  FP32; median of time - data_time over the iterations);
* parameters: work_dirs/eval/complexity.json (work_dirs/tmp/complexity.py);
* training curves: the per-evaluation tables in each run's log
  (tools/maritime_val_history.py -- the logged summary line of runs before
  2026-09-29 is a running mean over evaluations).
Re-run any time; pending evaluations are marked.
"""
import argparse
import csv
import glob
import json
import os
import pickle
import re
import statistics
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maritime_val_history import find_log, parse_log  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'projects', 'Maritime3D')
CLASSES = ['boat', 'buoy', 'sailboat', 'ship', 'yacht']
BANDS = ['0-50m', '50-100m', '100-160m']
SUBSETS = [('fov', 'camera FOV'), ('nonfov', 'outside FOV'), ('day', 'day'),
           ('night', 'night'), ('len0-15m', 'hull <15 m'),
           ('len15-30m', 'hull 15-30 m'), ('len>30m', 'hull >30 m')]

# key, run dir, eval name, display, modality, method, epochs, note
RUNS = [
    # --- baselines ----------------------------------------------------------
    ('gn_tfl', 'bench_gn_tfl', 'gn_tfl', 'TransFusion-L', 'L',
     'TransFusion head', 20, ''),
    ('gn_cp', 'bench_gn_cp', 'gn_cp', 'CenterPoint', 'L', 'CenterHead', 20,
     ''),
    ('gn_cp_80e', 'bench_gn_cp_80e', 'gn_cp_80e', 'CenterPoint', 'L',
     'CenterHead', 80, ''),
    ('gn_bevf_lc', 'bench_gn_bevf_lc', 'gn_bevf_lc', 'BEVFusion-LC', 'L+C',
     'Swin-T + LSS + TransFusion head', '20+6',
     'official recipe: fine-tuned from TransFusion-L (20 ep)'),
    ('gn_cp_bevfuse', 'bench_gn_cp_bevfuse', 'gn_cp_bevfuse',
     'CenterPoint + BEV-level fusion', 'L+C',
     'BEVFusion camera branch (Swin-T, LSS, ConvFuser) on CenterPoint',
     '20+6', 'fine-tuned from CenterPoint (20 ep), as MariFusion'),
    # --- ours: fusion -------------------------------------------------------
    ('gn_cp_refine_l', 'bench_gn_cp_refine_l', 'gn_cp_refine_l',
     'CenterPoint + 2nd stage, no camera', 'L',
     'ray-split refinement, LiDAR only', '20+6',
     'control for MariFusion: same stage and schedule without cameras'),
    ('gn_cp_marifusion', 'bench_gn_cp_marifusion', 'gn_cp_marifusion',
     'MariFusion', 'L+C', 'sparse instance fusion: reflection-aware '
     'sampling, reliability gate, ray-split refinement', '20+6',
     'fine-tuned from CenterPoint (20 ep)'),
    # --- ours: sea surface (training supervision) ---------------------------
    ('gn_cp_sea', 'bench_gn_cp_sea', 'gn_cp_sea', 'CenterPoint + Sea', 'L',
     'sea-surface supervision', 20, ''),
    ('gn_cp_sea_seatest', 'bench_gn_cp_sea', 'gn_cp_sea_seatest',
     'CenterPoint + Sea, sea also at test', 'L',
     'sea-surface supervision + test-time fusion', 20,
     'same checkpoint as CenterPoint + Sea'),
    ('gn_cp_sea_80e', 'bench_gn_cp_sea_80e', 'gn_cp_sea_80e',
     'CenterPoint + Sea', 'L', 'sea-surface supervision', 80,
     'stage 1 of the final model'),
    ('gn_cp_sea_80e_seatest', 'bench_gn_cp_sea_80e', 'gn_cp_sea_80e_seatest',
     'CenterPoint + Sea, sea also at test', 'L',
     'sea-surface supervision + test-time fusion', 80,
     'same checkpoint as CenterPoint + Sea (80 ep)'),
    # --- final model --------------------------------------------------------
    ('gn_cp_sea_marifusion', 'bench_gn_cp_sea_marifusion',
     'gn_cp_sea_marifusion', 'Ours: Sea + MariFusion', 'L+C',
     'sea-surface supervision + sparse instance fusion', '80+6',
     'fine-tuned from CenterPoint + Sea (80 ep)'),
    ('gn_cp_sea_marifusion_seatest', 'bench_gn_cp_sea_marifusion',
     'gn_cp_sea_marifusion_seatest', 'Ours, sea also at test', 'L+C',
     'as Ours + test-time sea-surface fusion', '80+6',
     'same checkpoint as Ours'),
    # --- ours on CenterFormer -----------------------------------------------
    ('gn_cf_sea_80e', 'bench_gn_cf_sea_80e', 'gn_cf_sea_80e',
     'CenterFormer + Sea', 'L', 'CenterFormer with sea-surface supervision',
     80, 'stage 1 of the final model on CenterFormer'),
    ('gn_cf_sea_marifusion', 'bench_gn_cf_sea_marifusion',
     'gn_cf_sea_marifusion', 'Ours on CenterFormer: Sea + MariFusion', 'L+C',
     'CenterFormer, sea-surface supervision + sparse instance fusion',
     '80+6', 'fine-tuned from CenterFormer + Sea (80 ep)'),
    # --- negative results ---------------------------------------------------
    ('gn_cp_ea', 'bench_gn_cp_ea', 'gn_cp_ea', 'CenterPoint + EA', 'L',
     'evidence anchor (returns centroid) + BEV a2c vector', 20,
     'finds more objects (AP@20m up), loses centres (boat AP@2m 57->39)'),
    ('gn_cp_ea2', 'bench_gn_cp_ea2', 'gn_cp_ea2',
     'CenterPoint + EA, local a2c', 'L',
     'evidence anchor + a2c in hull fractions', 20,
     'decoding through predicted l, w, yaw adds their errors'),
    ('gn_cp_ea_sea', 'bench_gn_cp_ea_sea', 'gn_cp_ea_sea',
     'CenterPoint + EA + Sea', 'L', 'evidence anchor + sea supervision', 20,
     ''),
    ('gn_cp_ea_sea_seatest', 'bench_gn_cp_ea_sea', 'gn_cp_ea_sea_seatest',
     'CenterPoint + EA + Sea, sea also at test', 'L',
     'evidence anchor + sea supervision + test-time fusion', 20, ''),
    ('gn_ssr_dt', 'bench_gn_ssr_dt', 'gn_ssr_dt',
     'TransFusion-L + SSR, fitted tilt', 'L',
     'rigid shared plane fitted per frame, decoupled, temporal', 20, ''),
    ('gn_ssr_imu', 'bench_gn_ssr_imu_r2', 'gn_ssr_imu',
     'TransFusion-L + SSR, IMU tilt', 'L',
     'rigid shared plane, IMU tilt, decoupled, temporal', 20, ''),
    # --- BatchNorm runs (train/eval normalisation gap) ----------------------
    ('bn_tfl_cyclic', 'bench_tfl', None, 'TransFusion-L, cyclic lr 5e-4 (BN)',
     'L', '', 20, 'diverged at the 5e-4 peak; stopped at ep16'),
    ('bn_tfl', 'bench_tfl_v2', None, 'TransFusion-L (BN)', 'L', '', 20,
     'BN train/eval gap'),
    ('bn_wahead', 'bench_tfl_wahead', None, 'TF-L + VWA + SSR v1 (BN)', 'L',
     '', 20, 'pooled-feature plane prior drifted to -10.7 m in eval mode; '
     'stopped at ep15'),
    ('bn_vwa', 'bench_tfl_vwa', None, 'TF-L + VWA (BN)', 'L', '', 20,
     'visible-waterline anchor; stopped at ep6; worst BN gap (var ratio 13x)'),
    ('bn_ssr', 'bench_tfl_ssr', None, 'TF-L + SSR coupled (BN)', 'L', '', 20,
     'gradients through the shared plane flattened the boxes (h 0.82x); '
     'stopped at ep11'),
    ('bn_ssr_dt', 'bench_tfl_ssr_dt', None, 'TF-L + SSR decoupled (BN)', 'L',
     '', 20, 'BN train/eval gap'),
    ('gn_ssr_imu_crash', 'bench_gn_ssr_imu', None, 'TF-L + SSR IMU, run 1',
     'L', '', 20, 'crashed at ep5 (DDP: tilt_offset without gradient); '
     're-run as gn_ssr_imu'),
]
BN_KEYS = [r[0] for r in RUNS if r[2] is None]

MAIN = ['gn_tfl', 'gn_cp', 'gn_cp_80e', 'gn_bevf_lc', 'gn_cp_bevfuse',
        'gn_cp_marifusion', 'gn_cp_sea_80e', 'gn_cp_sea_marifusion',
        'gn_cf_sea_80e', 'gn_cf_sea_marifusion']
OURS = 'gn_cp_sea_marifusion'


def eval_metrics(name, split):
    """Summary metrics of one evaluation, or None."""
    logs = sorted(glob.glob(os.path.join(ROOT, 'work_dirs', 'eval', name,
                                         split, '*', '*.log')))
    for log in reversed(logs):
        for line in open(log, errors='replace'):
            if 'Epoch(test)' in line and 'Maritime/' in line:
                return {k: float(v) for k, v in re.findall(
                    r'Maritime/(\S+): (-?[\d.]+(?:e-?\d+)?|nan)', line)}
    return None


def eval_speed(name):
    """(ms per frame, peak MB) over the val + test evaluation logs."""
    t, mem = [], []
    for log in glob.glob(os.path.join(ROOT, 'work_dirs', 'eval', name, '*',
                                      '*', '*.log')):
        rows = re.findall(r'Epoch\(test\).*time: ([\d.]+)\s+data_time: '
                          r'([\d.]+)\s+memory: (\d+)', open(
                              log, errors='replace').read())
        for a, b, m in rows[2:]:  # skip the warm-up iterations
            t.append(float(a) - float(b))
            mem.append(int(m))
    if len(t) < 3:
        return None
    return statistics.median(t) * 1000, max(mem)


def train_curve(run_dir):
    log = find_log(os.path.join(ROOT, 'work_dirs', run_dir))
    if not log:
        return {}
    return {k: v for k, v in parse_log(log).items() if k != 'test'}


def f(x, pct=True, nd=2):
    if x is None or x != x:
        return '–'
    return f'{x * 100:.{nd}f}' if pct else f'{x:.{nd}f}'


def d(a, b, pct=True):
    if a is None or b is None or a != a or b != b:
        return ''
    return '%+.2f' % ((a - b) * (100 if pct else 1))


def band_map(m, band):
    vals = [m.get(f'{c}_AP3D_{band}') for c in CLASSES]
    vals = [v for v in vals if v is not None and v == v]
    return sum(vals) / len(vals) if vals else None


def dataset_stats():
    cache = os.path.join(ROOT, 'work_dirs', 'eval', 'dataset_stats.json')
    if os.path.exists(cache):
        return json.load(open(cache))
    stats = {}
    for sp in ('train', 'val', 'test'):
        p = os.path.join(ROOT, 'dataset', 'infos', 'detection',
                         f'maritime_infos_{sp}.pkl')
        dl = pickle.load(open(p, 'rb'))['data_list']
        box, inst = Counter(), defaultdict(set)
        dn = Counter(f.get('daynight', '?') for f in dl)
        seqs = Counter(f.get('seq', '?') for f in dl)
        nonempty = sum(1 for f in dl if f['instances'])
        for fr in dl:
            for i in fr['instances']:
                c = CLASSES[i['bbox_label_3d']]
                box[c] += 1
                inst[c].add(i.get('instance_token'))
        stats[sp] = dict(frames=len(dl), nonempty=nonempty, day=dn['day'],
                         night=dn['night'], seqs=dict(seqs),
                         boxes={c: box[c] for c in CLASSES},
                         objects={c: len(inst[c]) for c in CLASSES})
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(stats, open(cache, 'w'), indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stdout', action='store_true')
    args = ap.parse_args()

    cplx_file = os.path.join(ROOT, 'work_dirs', 'eval', 'complexity.json')
    cplx = json.load(open(cplx_file)) if os.path.exists(cplx_file) else {}
    res = {}
    for (key, run, ev, disp, mod, method, ep, note) in RUNS:
        curve = train_curve(run)
        res[key] = dict(
            run=run, display=disp, modality=mod, method=method, epochs=ep,
            note=note, curve=curve,
            last_epoch=max(curve) if curve else None,
            val=eval_metrics(ev, 'val') if ev else None,
            test=eval_metrics(ev, 'test') if ev else None,
            speed=eval_speed(ev) if ev else None,
            params=cplx.get(key))

    def name(k, bold_ours=True):
        r = res[k]
        s = f"{r['display']} ({r['epochs']} ep)"
        return f'**{s}**' if bold_ours and k == OURS else s

    def m(k, split):
        return res[k][split] or {}

    L = []
    w = L.append
    w('# Maritime3D benchmark: experiment results')
    w('')
    w('Generated by `tools/collect_bench_results.py`; re-run to refresh. '
      'AP in %, AP40; errors in m / deg.')
    w('')
    w('**Protocol.** Official detection splits (train / val / test). IoU '
      'thresholds: boat, sailboat, ship, yacht 0.5; buoy 0.25 (3D and BEV). '
      'Range <= 160 m. mAPdist: BEV centre distance at 2 / 5 / 10 / 20 m, '
      'averaged. Every model: final epoch (no checkpoint selection on val), '
      'one standalone evaluation per split (`work_dirs/eval/`). All LiDAR '
      'backbones use GroupNorm in the 2D BEV backbone / neck (BatchNorm runs: '
      'Appendix C).')
    w('')
    w('**Ours** = CenterPoint trained with the physical sea surface as '
      'supervision (IMU tilt with learned gain, Kalman mean level, '
      'Gaussian-process waves, per-detection waterline uncertainty; 80 ep), '
      'then MariFusion (two front cameras, sparse instance-level fusion; '
      '+6 ep). At test time the detections keep the head\'s own heights; '
      'fusing them with the sea surface at test time is evaluated as an '
      'ablation ("sea also at test").')
    w('')
    w('**Epochs.** Baselines and ablations use 20 epochs (+6 for the camera '
      'stage); CenterPoint and the final model also 80. Compare rows with '
      'the same schedule; CenterPoint 20 -> 80 ep alone is +13.5 test mAP3D.')
    w('')
    w('**Val vs test.** Val holds 4 ships and 3 yachts (test: more of both); '
      'their AP swings by tens of points between evaluations of the same '
      'run, which moves val mAP by several points. Test is the reference.')
    w('')

    # ------------------------------------------------------------ table 1
    for split, title in (('test', 'Test'), ('val', 'Validation')):
        w(f'## Table 1{"a" if split == "test" else "b"}. Main results '
          f'({title} set)')
        w('')
        w('| Method | Mod. | Ep | mAP3D | mAPBEV | mAPdist | ATE-xy | ATE-z '
          '| mAOE | boat | buoy | sailboat | ship | yacht |')
        w('|' + '---|' * 14)
        for k in MAIN:
            r, x = res[k], m(k, split)
            if not x:
                w(f"| {name(k)} | {r['modality']} | {r['epochs']} | pending |"
                  + ' |' * 10)
                continue
            b = '**' if k == OURS else ''
            w(f"| {name(k)} | {r['modality']} | {r['epochs']} | "
              f"{b}{f(x.get('mAP3D'))}{b} | {b}{f(x.get('mAPBEV'))}{b} | "
              f"{f(x.get('mAPdist'))} | {f(x.get('ATE_xy'), False)} | "
              f"{f(x.get('ATE_z'), False)} | {f(x.get('mAOE'), False, 1)} | "
              + ' | '.join(f(x.get(f'{c}_AP3D')) for c in CLASSES) + ' |')
        w('')
        w('Per-class columns: AP@3D.')
        w('')

    # ------------------------------------------------------------ table 2
    w('## Table 2. Ablation: sea-surface supervision')
    w('')
    w('Same detector and schedule within each block; Δ against the first '
      'row of the block. "sea also at test": the same checkpoint with the '
      'detections\' bottoms fused with the sea surface at test time.')
    w('')
    w('| Variant | Ep | test mAP3D | test mAPBEV | Δ test 3D | test ATE-z | '
      'ATE-z hull >30 m | val mAP3D | val mAPBEV | val ATE-z |')
    w('|' + '---|' * 10)
    blocks = [['gn_cp', 'gn_cp_sea', 'gn_cp_sea_seatest'],
              ['gn_cp_80e', 'gn_cp_sea_80e', 'gn_cp_sea_80e_seatest'],
              ['gn_cp_sea_marifusion', 'gn_cp_sea_marifusion_seatest']]
    for bi, keys in enumerate(blocks):
        base = m(keys[0], 'test')
        for k in keys:
            t, v = m(k, 'test'), m(k, 'val')
            w(f"| {name(k)} | {res[k]['epochs']} | {f(t.get('mAP3D'))} | "
              f"{f(t.get('mAPBEV'))} | "
              f"{d(t.get('mAP3D'), base.get('mAP3D')) if k != keys[0] else ''}"
              f" | {f(t.get('ATE_z'), False)} | "
              f"{f(t.get('ATE_z_len>30m'), False)} | {f(v.get('mAP3D'))} | "
              f"{f(v.get('mAPBEV'))} | {f(v.get('ATE_z'), False)} |")
        if bi < len(blocks) - 1:
            w('| | | | | | | | | | |')
    w('')

    # ------------------------------------------------------------ table 3
    w('## Table 3. Ablation: camera fusion')
    w('')
    w('All camera variants start from the same LiDAR checkpoint and train '
      '6 more epochs. FOV: box centre projects into either front camera '
      '(3D AP on the test set). Parameters and latency: Table 8.')
    w('')
    w('| Variant | Mod. | Ep | test mAP3D | test mAPBEV | Δ test 3D | in FOV | '
      'outside FOV | yacht | val mAP3D | val mAPBEV |')
    w('|' + '---|' * 11)
    blocks = [['gn_cp', 'gn_cp_refine_l', 'gn_cp_bevfuse', 'gn_cp_marifusion'],
              ['gn_cp_sea_80e', 'gn_cp_sea_marifusion'],
              ['gn_cf_sea_80e', 'gn_cf_sea_marifusion']]
    for bi, keys in enumerate(blocks):
        base = m(keys[0], 'test')
        for k in keys:
            t, v = m(k, 'test'), m(k, 'val')
            w(f"| {name(k)} | {res[k]['modality']} | {res[k]['epochs']} | "
              f"{f(t.get('mAP3D'))} | {f(t.get('mAPBEV'))} | "
              f"{d(t.get('mAP3D'), base.get('mAP3D')) if k != keys[0] else ''}"
              f" | {f(t.get('mAP3D_fov'))} | {f(t.get('mAP3D_nonfov'))} | "
              f"{f(t.get('yacht_AP3D'))} | {f(v.get('mAP3D'))} | "
              f"{f(v.get('mAPBEV'))} |")
        if bi < len(blocks) - 1:
            w('|' + ' |' * 11)
    w('')

    # ------------------------------------------------------------ table 4
    w('## Table 4. Per-class AP@3D / AP@BEV (test set)')
    w('')
    w('| Method | ' + ' | '.join(CLASSES) + ' |')
    w('|' + '---|' * (len(CLASSES) + 1))
    for k in MAIN:
        x = m(k, 'test')
        if x:
            w(f'| {name(k)} | ' + ' | '.join(
                f"{f(x.get(f'{c}_AP3D'))} / {f(x.get(f'{c}_APBEV'))}"
                for c in CLASSES) + ' |')
    w('')

    # ------------------------------------------------------------ table 5
    w('## Table 5. AP@3D by distance (test set)')
    w('')
    w('mAP over the classes present in each band (ship and yacht have no '
      'object within 50 m in val / test).')
    w('')
    w('| Method | ' + ' | '.join(f'mAP {b}' for b in BANDS) + ' | ' +
      ' | '.join(f'boat {b}' for b in BANDS) + ' |')
    w('|' + '---|' * (1 + 2 * len(BANDS)))
    for k in MAIN:
        x = m(k, 'test')
        if x:
            w(f'| {name(k)} | ' + ' | '.join(
                f(band_map(x, b)) for b in BANDS) + ' | ' + ' | '.join(
                f(x.get(f'boat_AP3D_{b}')) for b in BANDS) + ' |')
    w('')

    # ------------------------------------------------------------ table 6
    for split, tag in (('test', 'a'), ('val', 'b')):
        w(f'## Table 6{tag}. Breakdowns, mAP@3D / mAP@BEV '
          f'({split} set)')
        w('')
        if split == 'test':
            w('Length bins use KITTI-style ignore (GT and unmatched '
              'detections outside the bin are ignored).')
            w('')
        w('| Method | ' + ' | '.join(n for _, n in SUBSETS) + ' |')
        w('|' + '---|' * (len(SUBSETS) + 1))
        for k in MAIN:
            x = m(k, split)
            if x:
                w(f'| {name(k)} | ' + ' | '.join(
                    f"{f(x.get(f'mAP3D_{s}'))} / {f(x.get(f'mAPBEV_{s}'))}"
                    for s, _ in SUBSETS) + ' |')
        w('')

    # ------------------------------------------------------------ table 7
    w('## Table 7. Localisation and orientation errors (test set)')
    w('')
    w('ATE over detections matched within 5 m of BEV centre distance (m); '
      'mAOE: mean absolute yaw error within 10 m (deg).')
    w('')
    lens = ['len0-15m', 'len15-30m', 'len>30m']
    w('| Method | ATE-xy | ATE-z | ATE-z median | ' + ' | '.join(
        f'ATE-z {l[3:]}' for l in lens) + ' | mAOE |')
    w('|' + '---|' * (5 + len(lens)))
    for k in MAIN[:6] + ['gn_cp_sea_80e', 'gn_cp_sea_80e_seatest', OURS,
                         'gn_cp_sea_marifusion_seatest']:
        x = m(k, 'test')
        if x:
            w(f"| {name(k)} | {f(x.get('ATE_xy'), False)} | "
              f"{f(x.get('ATE_z'), False)} | "
              f"{f(x.get('ATE_z_median'), False)} | " + ' | '.join(
                  f(x.get(f'ATE_z_{l}'), False) for l in lens) +
              f" | {f(x.get('mAOE'), False, 1)} |")
    w('')

    # ------------------------------------------------------------ table 8
    w('## Table 8. Model complexity')
    w('')
    w('Parameters counted on the built model. Latency: median inference '
      'time per frame in the val / test evaluations (A800, 1 frame per GPU, '
      'FP32, data loading excluded, post-processing included); memory: '
      'peak allocated. The sea surface adds 10 scalars and a closed-form '
      'Kalman + GP solve in training only; its 0.57 M are the extra '
      'waterline-uncertainty / freeboard head branches (and an unused '
      'anchor-to-centre branch, 0.19 M, kept for checkpoint compatibility).')
    w('')
    w('| Method | Mod. | Params (M) | LiDAR enc. | BEV 2D | head | image '
      'branch | LSS + fuser | refine head | ms / frame | memory (MB) |')
    w('|' + '---|' * 11)
    for k in MAIN[:2] + ['gn_cp_sea'] + MAIN[3:6] + ['gn_cp_refine_l', OURS,
                                                      'gn_cf_sea_80e',
                                                      'gn_cf_sea_marifusion']:
        p, s = res[k]['params'] or {}, res[k]['speed']
        g = p.get('groups', {})
        w(f"| {res[k]['display']} | {res[k]['modality']} | "
          f"{f(p.get('total'), False)} | " + ' | '.join(
              f(g.get(x), False) for x in ('lidar_enc', 'bev_2d', 'head',
                                           'img', 'lss+fuse', 'refine')) +
          f" | {f(s[0], False, 1) if s else '–'} | "
          f"{s[1] if s else '–'} |")
    w('')

    # ------------------------------------------------------- appendix A
    w('## Appendix A. Negative results')
    w('')
    w('| Variant | Ep | test mAP3D | test mAPBEV | val mAP3D | val mAPBEV | '
      'boat (test) | ship (test) | Finding |')
    w('|' + '---|' * 9)
    for k in ['gn_cp', 'gn_cp_ea', 'gn_cp_ea2', 'gn_cp_ea_sea',
              'gn_cp_ea_sea_seatest', 'gn_tfl', 'gn_ssr_dt', 'gn_ssr_imu']:
        t, v = m(k, 'test'), m(k, 'val')
        w(f"| {res[k]['display']} | {res[k]['epochs']} | "
          f"{f(t.get('mAP3D'))} | {f(t.get('mAPBEV'))} | "
          f"{f(v.get('mAP3D'))} | {f(v.get('mAPBEV'))} | "
          f"{f(t.get('boat_AP3D'))} | {f(t.get('ship_AP3D'))} | "
          f"{res[k]['note'] or res[k]['method']} |")
    w('')
    w('* Evidence anchor (EA): the heatmap peak at the centroid of each '
      'hull\'s own returns. It recovers more objects (BEV-distance AP@20m '
      'up for boat / buoy / sailboat) but the anchor-to-centre regression '
      'loses more centres than that gains, in either parameterisation.')
    w('* Rigid shared sea plane (SSR) on TransFusion-L: a single plane per '
      'frame, fitted (SSR fitted tilt) or tilted by the IMU; superseded by '
      'the physical sea surface (Kalman level + GP waves) on CenterPoint.')
    w('')

    # ------------------------------------------------------- appendix B
    w('## Appendix B. Training curves (val, every evaluation)')
    w('')
    w('True per-evaluation values recovered from the tables the metric '
      'prints (`tools/maritime_val_history.py`): mmengine averaged every '
      'logged scalar matching `.*(loss|time|...).*` and "Mari*time*/mAP3D" '
      'matched, so the val summary lines in the logs of runs before '
      '2026-09-29 are running means. The sea-surface runs were evaluated '
      'during training with the sea surface also at test.')
    w('')
    cols = ['mAP3D', 'mAPBEV', 'mAPdist'] + [f'{c}_AP3D' for c in CLASSES]
    seen = set()
    for (key, run, *_rest) in RUNS:
        r = res[key]
        if not r['curve'] or run in seen:
            continue
        seen.add(run)
        extra = f" — {r['note']}" if r['note'] and key in BN_KEYS else ''
        w(f"**{r['display']} ({r['epochs']} ep)** (`{run}`){extra}")
        w('')
        w('| ep | ' + ' | '.join(cols) + ' |')
        w('|' + '---|' * (len(cols) + 1))
        for e in sorted(r['curve']):
            v = r['curve'][e]
            w(f'| {e} | ' + ' | '.join(
                '–' if v.get(c) is None else f'{v[c]:.2f}' for c in cols)
              + ' |')
        w('')

    # ------------------------------------------------------- appendix C
    w('## Appendix C. BatchNorm runs and stopped runs')
    w('')
    w('With 2 frames per GPU and scenes from open water to crowded '
      'harbours, the BatchNorm statistics of the 2D BEV backbone were '
      'per-frame in training but population averages at test time '
      '(running mean off by up to 1.6 sd); boat AP3D roughly doubled with '
      'per-frame statistics at inference. Every run in the tables above '
      'uses GroupNorm there instead.')
    w('')
    w('| Run | Method | Why not in the tables |')
    w('|---|---|---|')
    for k in BN_KEYS:
        w(f"| `{res[k]['run']}` | {res[k]['display']} | {res[k]['note']} |")
    w('')

    # ------------------------------------------------------- appendix D
    st = dataset_stats()
    w('## Appendix D. Dataset statistics (detection splits)')
    w('')
    w('| split | frames | with objects | day / night | ' + ' | '.join(
        f'{c} boxes (objects)' for c in CLASSES) + ' |')
    w('|' + '---|' * (4 + len(CLASSES)))
    for sp in ('train', 'val', 'test'):
        s = st[sp]
        w(f"| {sp} | {s['frames']} | {s['nonempty']} | {s['day']} / "
          f"{s['night']} | " + ' | '.join(
              f"{s['boxes'][c]} ({s['objects'][c]})" for c in CLASSES) + ' |')
    w('')
    w('Note: the same physical vessels appear in several splits for ship and '
      'yacht (e.g. ship c0b9766a0f and f804695e7b are in train, val and '
      'test); val / test ship and yacht AP measure re-identification of '
      'seen hulls as much as generalisation.')

    md = '\n'.join(L) + '\n'
    open(os.path.join(OUT, 'RESULTS_BENCH.md'), 'w').write(md)
    json.dump(res, open(os.path.join(OUT, 'results_bench.json'), 'w'),
              indent=1, default=str)
    with open(os.path.join(OUT, 'results_bench.csv'), 'w', newline='') as fh:
        keys = sorted({kk for r in res.values() for s in ('val', 'test')
                       if r[s] for kk in r[s]})
        wr = csv.writer(fh)
        wr.writerow(['key', 'run', 'display', 'modality', 'epochs', 'split']
                    + keys)
        for k, r in res.items():
            for s in ('val', 'test'):
                if r[s]:
                    wr.writerow([k, r['run'], r['display'], r['modality'],
                                 r['epochs'], s] +
                                [r[s].get(kk, '') for kk in keys])
    if args.stdout:
        print(md)


if __name__ == '__main__':
    main()
