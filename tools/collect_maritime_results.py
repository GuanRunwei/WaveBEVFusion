"""Collect maritime3d baseline results into one markdown table.

Reads the final metric line out of each ``work_dirs/<name>.test.log`` and prints
a benchmark summary. Run after tools/run_maritime_benchmark.sh finishes:

    python tools/collect_maritime_results.py > projects/Maritime3D/RESULTS.md

If ``work_dirs/ci_<name>.json`` exists (written by
``tools/bootstrap_maritime_ci.py``) the encounter-level confidence intervals are
folded in, which is the only part of this table that says whether two methods
actually differ.
"""
import argparse
import json
import os
import re
import sys

# display name -> work_dirs stem, in the order they should appear in the table
BASELINES = [
    ('PointPillars', 'pointpillars', 'LiDAR'),
    ('SECOND', 'second', 'LiDAR'),
    ('TransFusion-L', 'transfusion_lidar', 'LiDAR'),
    ('BEVFusion', 'bevfusion_lidar-cam', 'LiDAR + stereo camera'),
    ('PETR', 'petr', 'stereo camera'),
]
CLASSES = ['boat', 'ship', 'sailboat', 'buoy']
RANGES = ['0-50m', '50-100m', '100-160m']
DISTS = ['2m', '5m', '10m', '20m']
STATES = ['calm', 'moderate', 'rough']
METRIC_RE = re.compile(r'(\w+)/([\w.\-]+):\s*(nan|[-\d.eE+]+)')


def parse_log(path):
    """Return {prefix: {metric: value}} from the last metric line in a log."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path) as f:
        for line in f:
            if 'Epoch(test)' in line and 'AP3D' in line:
                last = line
    if last is None:
        return None
    out = {}
    for prefix, key, val in METRIC_RE.findall(last):
        out.setdefault(prefix, {})[key] = float(val)
    return out


def fmt(value):
    if value is None:
        return '--'
    return '--' if value != value else f'{value * 100:.2f}'


def deg(value):
    """Angles are already in degrees, unlike the APs, which are fractions."""
    if value is None:
        return '--'
    return '--' if value != value else f'{value:.1f}'


def degenerate(res):
    """True if every reported AP is exactly zero.

    A DETR-style head whose classifier has collapsed to a constant score emits
    boxes but ranks them arbitrarily, so AP40 is 0 everywhere regardless of
    localisation quality. That is a failed run, not a measurement, and printing
    it as ``0.00`` alongside trained baselines would misrepresent it.

    Only AP keys count. Orientation error is a magnitude, not a score: a run
    that detects nothing still reports some yaw error on whatever it matched,
    and letting that non-zero value into the test would mask the collapse.
    """
    if not res:
        return False
    vals = [v for m in res.values() for k, v in m.items()
            if v == v and ('AP' in k or k.startswith('mAP'))]
    return bool(vals) and all(v == 0.0 for v in vals)


def load_ci(work_dir, stem):
    path = os.path.join(work_dir, f'ci_{stem}.json')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def ci_section(work_dir, failed):
    """Encounter-level confidence intervals, if the bootstrap has been run."""
    have = [(n, s) for n, s, _ in BASELINES
            if s not in failed and load_ci(work_dir, s)]
    if not have:
        return
    cis = {s: load_ci(work_dir, s) for _, s in have}
    enc = next(iter(cis.values()))['classes']

    print('\n## Confidence intervals\n')
    print('95% intervals from `tools/bootstrap_maritime_ci.py`, which resamples')
    print('**independent object encounters** with replacement, not frames. At')
    print('10 Hz the 130 test `ship` boxes are one vessel drifting 0.10 m per')
    print('frame; treating them as 130 draws would understate the interval by')
    print('roughly the square root of that duplication.\n')
    head = ['Method'] + [f'{c} ({enc[c]["encounters"]} enc)' for c in CLASSES]
    rows = [head, ['---'] * len(head)]
    for name, stem in have:
        cells = []
        for c in CLASSES:
            e = cis[stem]['classes'][c]
            if 'lo' in e:
                cells.append(f'{e["point"] * 100:.1f} '
                             f'[{e["lo"] * 100:.1f}, {e["hi"] * 100:.1f}]')
            else:
                cells.append(f'{e["point"] * 100:.1f} (n/a)')
        rows.append([name] + cells)
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')
    print('\n`(n/a)` means the class has too few independent encounters for a')
    print('bootstrap to mean anything -- with 4 encounters there are only 4**4 =')
    print('256 distinct resamples, and the percentiles are an artefact of that')
    print('support rather than a statement about the method. Only `boat`')
    print('qualifies, and its interval is wide enough to overlap across every')
    print('baseline in this table. **No pair of methods here is separated by')
    print('this test split.**')


def dist_section(results, failed):
    """Centre-distance AP, if the logs are new enough to carry it."""
    have = [(n, s) for n, s, _ in BASELINES
            if s not in failed
            and (results[s] or {}).get('Maritime', {}).get('mAPdist') is not None]
    if not have:
        return
    print('\n## AP@3D by BEV centre distance\n')
    print('Matches a detection to ground truth by centre distance instead of')
    print('IoU, so extent and heading are ignored: this answers "is there a')
    print('vessel about there", not "is this box right". Reported because')
    print('IoU 0.5 on a 6 m boat demands ~2 m of centre accuracy, which a')
    print('0.84 m stereo baseline cannot deliver at 50-160 m -- every AP@IoU a')
    print('camera-only method produces is exactly 0, and a column of zeros')
    print('cannot distinguish "learned nothing" from "off by 10 m". nuScenes')
    print('uses centre distance for the same reason.\n')
    head = ['Method', 'Modality'] + [f'mAP@{d}' for d in DISTS] + ['mean']
    rows = [head, ['---'] * len(head)]
    for name, stem in have:
        r = results[stem]['Maritime']
        modality = next(m for n, s, m in BASELINES if s == stem)
        per_d = []
        for d in DISTS:
            vals = [r.get(f'{c}_APdist_{d}') for c in CLASSES]
            vals = [v for v in vals if v is not None and v == v]
            per_d.append(fmt(sum(vals) / len(vals) if vals else None))
        rows.append([name, modality] + per_d + [fmt(r.get('mAPdist'))])
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


def orientation_section(results, failed):
    """Yaw error, if the logs are new enough to carry it."""
    have = [(n, s) for n, s, _ in BASELINES
            if s not in failed
            and (results[s] or {}).get('Maritime', {}).get('mAOE') is not None]
    if not have:
        return
    print('\n## Orientation error (yaw)\n')
    print('Mean absolute heading error, in degrees, over detections matched to')
    print('ground truth within 10 m of BEV centre distance -- matching on IoU')
    print('would measure yaw only where the box is already good. `AOE` counts a')
    print('bow/stern swap as 180 deg; `AOE180` folds the angle to [0, 90] and')
    print('does not, so a gap between the two columns means the heading *axis*')
    print('is right and the direction along it is not.\n')
    print('Yaw is the only Euler angle scored here. See README section 4 on why')
    print('pitch and roll are not, and what that omits.\n')
    head = ['Method'] + [f'{c} AOE' for c in CLASSES] + ['mAOE', 'mAOE180']
    rows = [head, ['---'] * len(head)]
    for name, stem in have:
        r = results[stem]['Maritime']
        f180 = [r.get(f'{c}_AOE180') for c in CLASSES]
        f180 = [v for v in f180 if v is not None and v == v]
        rows.append([name] +
                    [deg(r.get(f'{c}_AOE')) for c in CLASSES] +
                    [deg(r.get('mAOE')),
                     deg(sum(f180) / len(f180) if f180 else None)])
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')


def sea_state_section(results, failed):
    """AP@3D against how hard the hull was rocking, if the logs carry it."""
    have = [(n, s) for n, s, _ in BASELINES
            if s not in failed
            and (results[s] or {}).get('Maritime',
                                       {}).get('mAP3D_calm') is not None]
    if not have:
        return
    print('\n## AP@3D by sea state\n')
    print('Test frames banded by how hard the hull is rocking -- the RMS of its')
    print('roll+pitch rate over a 4 s window, from')
    print('`data/maritime/ego_attitude.npz`: calm < 0.77 deg/s, moderate')
    print('0.77-1.48 deg/s, rough > 1.48 deg/s (1174 / 1623 / 1087 test')
    print('frames). Edges are the terciles of the full annotated record, fixed')
    print('as constants so the bands stay comparable as sequences are added.\n')
    print('This is the Euler-angle question the data can answer. Per-object')
    print('pitch and roll are unannotated, but the *platform* rolls at 0.70 deg')
    print('RMS on a 2.25 s period and pitches at 1.35 deg RMS on a 19 s swell,')
    print('and that is what makes a level-ground prior wrong: one degree of')
    print('tilt displaces a target 2.6 m at 150 m, against an IoU-0.5 budget of')
    print('about 2 m for a 6 m boat.\n')
    print('**Rate, not tilt.** An accelerometer cannot separate a roll from the')
    print('centripetal acceleration of a sustained turn, so banding on tilt')
    print('bands partly on turning -- and turning is not independent of what is')
    print('in the scene. The tilt-banded version of this table put 11233 boat')
    print('boxes at a median 84 m in calm against 194 at a median 14 m in')
    print('rough, which compares different vessels, not different conditions.')
    print('A rate RMS drops both a steady turn and a static trim (near-DC in')
    print('rate), leaving the wave-driven oscillation, and the bands below are')
    print('matched at median boat range 82 / 81 / 78 m with 33 / 27 / 37 per')
    print('cent of boats beyond 100 m over 21 / 26 / 23 temporal blocks.\n')
    head = ['Method'] + [f'boat {s}' for s in STATES] + \
        ['boat delta'] + [f'mAP {s}' for s in STATES]
    rows = [head, ['---'] * len(head)]
    for name, stem in have:
        r = results[stem]['Maritime']
        calm, rough = r.get('boat_AP3D_calm'), r.get('boat_AP3D_rough')
        drop = (None if calm is None or rough is None
                or calm != calm or rough != rough else rough - calm)
        rows.append([name] +
                    [fmt(r.get(f'boat_AP3D_{s}')) for s in STATES] +
                    ['--' if drop is None else f'{drop * 100:+.2f}'] +
                    [fmt(r.get(f'mAP3D_{s}')) for s in STATES])
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')
    print('\n`boat` carries this table and the `mAP` columns should be ignored')
    print('in it. `ship` has no encounter at all in the rough band and `buoy`')
    print('none in the calm one, so their AP there is `nan` and the mean')
    print('silently becomes an average over whichever classes happened to')
    print('survive. `boat` is the only class with enough independent')
    print('encounters (12) to band three ways at all, and even it is thin: the')
    print('rough band holds 2233 boat boxes but they are not 2233 independent')
    print('observations.\n')
    print('**Read as a null result.** No method degrades monotonically with sea')
    print('state, the `boat delta` changes sign across methods, and all four')
    print('dip in the *moderate* band and recover in the rough one -- a shape')
    print('no motion mechanism explains, and one that points at scene')
    print('composition rather than at waves. The deltas are also far inside the')
    print('~19-point `boat` interval from the encounter bootstrap. On this')
    print('recording, residual sensitivity to hull rocking is not measurable at')
    print('this split\'s resolution -- which is consistent with the levelling')
    print('already applied upstream of the point cloud (README section 4), and')
    print('is a statement about this sequence, not about detectors at sea.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', default='work_dirs')
    args = parser.parse_args()

    results = {}
    for _, stem, _ in BASELINES:
        results[stem] = parse_log(os.path.join(args.work_dir,
                                               f'{stem}.test.log'))

    missing = [s for s, r in results.items() if r is None]
    if missing:
        print(f'<!-- no test log yet for: {", ".join(missing)} -->\n')
    failed = {s for s, r in results.items() if degenerate(r)}

    print('# maritime3d benchmark results\n')
    print('Sequence 00, held-out test split (3884 frames / 15354 objects).')
    print('AP40 at IoU 0.5 for boat / ship / sailboat and 0.25 for buoy.\n')

    # main table: mAP and per-class 3D AP
    head = ['Method', 'Modality', 'mAP@3D', 'mAP@BEV'] + \
        [f'{c} @3D' for c in CLASSES]
    rows = [head, ['---'] * len(head)]
    for name, stem, modality in BASELINES:
        if stem in failed:
            rows.append([name, modality] + ['FAILED'] + ['--'] * 5)
            continue
        r = (results[stem] or {}).get('Maritime', {})
        rows.append([name, modality,
                     fmt(r.get('mAP3D')), fmt(r.get('mAPBEV'))] +
                    [fmt(r.get(f'{c}_AP3D')) for c in CLASSES])
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    print('\n> **Do not rank methods on `mAP@3D`.** It averages four classes '
          'whose independent\n> support differs by an order of magnitude: the '
          'test split holds 12 boat encounters\n> but only 1 ship, 2 sailboat '
          'and 4 buoy. Three quarters of the mean therefore\n> rests on seven '
          'unique objects, and `ship AP` is the answer to "was this one '
          'vessel\n> found". See the confidence intervals below and README '
          'section 2.')

    ci_section(args.work_dir, failed)
    dist_section(results, failed)
    orientation_section(results, failed)
    sea_state_section(results, failed)

    for stem in failed:
        name = next(n for n, s, _ in BASELINES if s == stem)
        print(f'\n> **{name} did not converge and is not reported.** Every AP '
              'it produced is exactly\n'
              '> zero, which means no detection cleared the IoU gate on any '
              'class at any range --\n'
              '> a failed run, not a measurement, and printing it as `0.00` '
              'beside trained\n'
              '> baselines would misrepresent it. Diagnose with\n'
              '> `tools/probe_petr_box_quality.py`, which reports where the '
              'boxes actually land\n'
              '> instead of whether they cleared a threshold.')

    # range breakdown
    print('\n## AP@3D by range\n')
    head = ['Method'] + [f'{c} {b}' for c in CLASSES for b in RANGES]
    rows = [head, ['---'] * len(head)]
    for name, stem, _ in BASELINES:
        if stem in failed:
            continue
        r = (results[stem] or {}).get('Maritime', {})
        rows.append([name] + [
            fmt(r.get(f'{c}_AP3D_{b}')) for c in CLASSES for b in RANGES
        ])
    for row in rows:
        print('| ' + ' | '.join(row) + ' |')

    # the camera-only baseline also reports inside its own field of view
    fov = None if 'petr' in failed else (results['petr'] or {}).get(
        'MaritimeFOV')
    if fov:
        print('\n## Camera-sector protocol (PETR only)\n')
        print('PETR sees roughly [-35, +24] deg of the 360 deg the LiDAR is')
        print('annotated over, so the full-circle mAP above is bounded by the')
        print('rig, not the method. Scored inside the camera sector:\n')
        head = ['Metric'] + CLASSES + ['mAP']
        rows = [head, ['---'] * len(head)]
        for label, key in (('AP@3D', 'AP3D'), ('AP@BEV', 'APBEV')):
            rows.append([label] + [fmt(fov.get(f'{c}_{key}'))
                                   for c in CLASSES] +
                        [fmt(fov.get(f'm{key}'))])
        for row in rows:
            print('| ' + ' | '.join(row) + ' |')

    return 0 if not missing else 1


if __name__ == '__main__':
    sys.exit(main())
