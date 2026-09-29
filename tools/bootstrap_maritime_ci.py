"""Bootstrap confidence intervals for the maritime3d baselines.

The headline mAP averages four classes whose *independent* support differs by
more than an order of magnitude. Object counts are misleading: the test split
contains 130 "ship" objects, but they are one vessel tracked across 130
consecutive 10 Hz frames, drifting 0.1 m per frame. Counting encounters rather
than boxes:

    class      train  val  test
    boat          20    8    12
    ship           3    1     1
    sailboat       4    1     2
    buoy          10    2     4

So three of the four classes -- 75% of an unweighted mAP -- rest on seven unique
object encounters in the test split, and ship AP is literally "did the detector
find this one vessel". That is why BEVFusion and TransFusion-L swap places by 50
AP points on buoy between val and test while sitting within 3 points of each
other on boat, the only class with real support.

This tool therefore resamples **encounters**, not frames. A frame-level
bootstrap would treat 130 views of the same ship as 130 independent draws and
report a confidence interval far tighter than the truth. An encounter is a
maximal run of frames containing the class with no gap longer than
``--gap`` frames; resampling those with replacement is a block bootstrap, which
is the right unit when observations are serially correlated.

    python tools/bootstrap_maritime_ci.py --model transfusion_lidar --n 500

Reuses MaritimeMetric's own AP implementation, so the point estimate it prints
reproduces the number in the results table exactly.

Only ``boat`` clears ``--min-encounters``. That is the finding, not a limitation
of the tool: with four encounters a bootstrap draws from 4**4 = 256 distinct
resamples and its percentiles are an artefact of that tiny support -- at n=5 the
buoy interval did not even contain its own point estimate. For the other three
classes the tool prints the encounter count instead of inventing an interval.
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

sys.path.insert(0, os.getcwd())
from mmdet3d.registry import METRICS, MODELS  # noqa: E402

CFG_DIR = 'projects/Maritime3D/configs'


def frame_ids(cfg):
    """``sample_idx`` per test frame, in the order the loader yields them.

    The test split is 26 non-contiguous temporal blocks concatenated, so a gap
    measured in list position would silently weld the end of one block to the
    start of another and undercount encounters. Frame index in the original
    sequence is the only thing that says whether two frames are 0.1 s or
    2 minutes apart.
    """
    ann = os.path.join(cfg.test_dataloader.dataset.data_root,
                       cfg.test_dataloader.dataset.ann_file)
    with open(ann, 'rb') as f:
        return np.array([s['sample_idx'] for s in pickle.load(f)['data_list']])


def collect(model_stem, epoch, device):
    """Run the test split once and return the metric plus its raw per-frame."""
    cfg = Config.fromfile(f'{CFG_DIR}/{model_stem}_maritime-3d-4class.py')
    init_default_scope('mmdet3d')
    loader = Runner.build_dataloader(cfg.test_dataloader)

    metric_cfg = cfg.test_evaluator
    if isinstance(metric_cfg, (list, tuple)):
        metric_cfg = metric_cfg[0]
    metric = METRICS.build(dict(metric_cfg))
    metric.dataset_meta = loader.dataset.metainfo

    model = MODELS.build(cfg.model)
    load_checkpoint(model, f'work_dirs/{model_stem}/epoch_{epoch}.pth',
                    map_location='cpu')
    model.to(device).eval()

    with torch.no_grad():
        for data in loader:
            metric.process(data, [s.to_dict() for s in model.test_step(data)])
    return metric, frame_ids(cfg)


def encounters(prepared, sids, ci, gap):
    """Group frame positions into runs of frames containing class ``ci``.

    A run ends when more than ``gap`` frames of the original sequence separate
    two consecutive sightings. Each run is one independent encounter with (in
    practice) one vessel; the frames inside it are near-duplicate views of it.
    """
    hit = [i for i, f in enumerate(prepared) if (f['gt_labels'] == ci).any()]
    if not hit:
        return []
    runs, cur = [], [hit[0]]
    for a, b in zip(hit, hit[1:]):
        if sids[b] - sids[a] > gap:
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    return runs


def ap_over(metric, frames, ci, name):
    """AP@3D for one class over an explicit list of prepared frames."""
    from projects.Maritime3D.maritime3d.maritime_metric import _eval_class
    sel = []
    for f in frames:
        pm = f['labels'] == ci
        gm = f['gt_labels'] == ci
        sel.append(
            dict(boxes=f['boxes'][pm],
                 scores=f['scores'][pm],
                 gt=f['gt_boxes'][gm]))
    return _eval_class(sel, metric.iou_thresholds[name], '3d')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--epoch', type=int, default=40)
    p.add_argument('--n', type=int, default=500, help='bootstrap resamples')
    p.add_argument('--gap', type=int, default=15,
                   help='frames of absence that end an encounter (1.5 s)')
    p.add_argument('--min-encounters', type=int, default=8,
                   help='below this, report the count instead of an interval')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    metric, sids = collect(args.model, args.epoch, args.device)
    classes = list(metric.dataset_meta['classes'])

    # apply the range/azimuth filter once, exactly as compute_metrics does
    metric._prepared = []
    for r in metric.results:
        pm = metric._in_limit(r['boxes'])
        gm = metric._in_limit(r['gt_boxes'])
        metric._prepared.append(
            dict(boxes=r['boxes'][pm],
                 scores=r['scores'][pm],
                 labels=r['labels'][pm],
                 gt_boxes=r['gt_boxes'][gm],
                 gt_labels=r['gt_labels'][gm]))

    n_frames = len(metric._prepared)
    rng = np.random.default_rng(args.seed)
    report = {'model': args.model, 'n_frames': n_frames,
              'n_bootstrap': args.n, 'classes': {}}

    print(f'\n{args.model}: block bootstrap over test encounters '
          f'({n_frames} frames, {args.n} resamples)\n')
    print(f'{"class":10s} {"AP@3D":>7s} {"enc":>4s}  {"95% CI":>16s}')
    for ci, name in enumerate(classes):
        runs = encounters(metric._prepared, sids, ci, args.gap)
        point = ap_over(metric, metric._prepared, ci, name)
        entry = dict(point=float(point), encounters=len(runs))
        if len(runs) < args.min_encounters:
            # a bootstrap over 1-3 encounters resamples the same object and
            # would print a spuriously tight interval
            print(f'{name:10s} {point * 100:7.2f} {len(runs):4d}  '
                  f'{"not estimable":>16s}')
            report['classes'][name] = entry
            continue
        draws = []
        for _ in range(args.n):
            pick = rng.integers(0, len(runs), len(runs))
            frames = [metric._prepared[i] for k in pick for i in runs[k]]
            draws.append(ap_over(metric, frames, ci, name))
        d = np.asarray(draws, dtype=float)
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        print(f'{name:10s} {point * 100:7.2f} {len(runs):4d}  '
              f'[{lo * 100:6.2f}, {hi * 100:6.2f}]')
        entry.update(lo=float(lo), hi=float(hi))
        report['classes'][name] = entry

    pts = [report['classes'][c]['point'] for c in classes]
    print(f'\n{"mAP":10s} {np.nanmean(pts) * 100:7.2f}   '
          '(unweighted over 4 classes -- see the module docstring on why this '
          'is not a\n           rankable quantity on this data)')
    report['mAP'] = float(np.nanmean(pts))

    if args.out:
        with open(args.out, 'w') as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
