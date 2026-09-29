"""How much 3D IoU costs to ignore pitch and roll, per class, on this data.

Backs the argument in ``projects/Maritime3D/EULER_ANGLES.md``. Two parts:

1. **Geometry.** A 7-DoF detector emits a box with zero pitch and zero roll. If
   the target is actually tilted by theta, the best that detector can score --
   even with perfect centre, extent and yaw -- is the IoU between the tilted
   box and its un-tilted twin. That is a hard ceiling, computable from the
   class median dimensions alone, and it is what the ``crit`` column reports:
   the angle at which the ceiling falls to the class's evaluation threshold.

   The rotation is exact rather than sampled. A box is a prism, so rotating it
   about the lateral (y) axis leaves its y extent untouched and the 3D IoU
   collapses to the 2D IoU of the (dx, dz) cross-section. Roll about the
   longitudinal (x) axis likewise collapses to (dy, dz). Both are rectangle
   overlaps, and Sutherland-Hodgman clipping gives them to machine precision.

2. **Headroom.** The critical angles come out at 21-30 deg, which no real
   vessel reaches, and that reads as "7-DoF is fine". It is not, because a
   detector does not arrive at the threshold with the full budget in hand --
   it is already spending most of it on centre and extent error. Given dumped
   predictions, the second table reports the share of *currently passing*
   ground-truth boxes whose IoU would fall below the threshold if the target
   were tilted. On sequence 00 that is 25% of `boat` at 5 deg of pitch, against
   a critical angle of 29.8 deg. The tax is on the margin, not on the cliff.

Usage::

    python tools/analyze_attitude_sensitivity.py                # geometry
    python tools/analyze_attitude_sensitivity.py --preds p.pkl  # + headroom

The prediction pkl is whatever ``mmengine.evaluator.DumpResults`` writes; add
it to a config's ``test_evaluator`` list and run ``tools/test.py`` as usual.

Caveat, and it is the important one: no target vessel's attitude is annotated
anywhere in this dataset, so nothing here measures how much the *targets*
actually pitch and roll. Every angle is a counterfactual -- "if a vessel were
heeled this far, this is what it would cost". The only measured attitude is the
observer's, from the IMU, and ``tools/extract_ego_attitude.py`` handles that.
"""
import argparse
import os
import pickle
import sys

import numpy as np

CLASSES = ('boat', 'ship', 'sailboat', 'buoy')
THRESHOLDS = {'boat': 0.5, 'ship': 0.5, 'sailboat': 0.5, 'buoy': 0.25}
ANGLES = (2.0, 5.0, 10.0)


def _clip(subject, clipper):
    """Sutherland-Hodgman: part of ``subject`` inside convex ``clipper``."""
    out = list(subject)
    for i in range(len(clipper)):
        p1, p2 = clipper[i], clipper[(i + 1) % len(clipper)]
        inp, out = out, []
        if not inp:
            break

        def side(p, p1=p1, p2=p2):
            return ((p2[0] - p1[0]) * (p[1] - p1[1]) -
                    (p2[1] - p1[1]) * (p[0] - p1[0]))

        s = inp[-1]
        for e in inp:
            if side(e) >= 0:
                if side(s) < 0:
                    t = side(s) / (side(s) - side(e))
                    out.append(s + t * (e - s))
                out.append(e)
            elif side(s) >= 0:
                t = side(s) / (side(s) - side(e))
                out.append(s + t * (e - s))
            s = e
    return np.asarray(out)


def _area(poly):
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def ceiling(a, b, deg):
    """IoU between an ``a x b`` box and the same box rotated by ``deg``.

    The ceiling on what a detector that never predicts this angle can score
    against a target that has it, with everything else perfect.
    """
    r = np.array([[-a / 2, -b / 2], [a / 2, -b / 2],
                  [a / 2, b / 2], [-a / 2, b / 2]])
    t = np.radians(deg)
    rot = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    inter = _area(_clip(r, r @ rot.T))
    return inter / (2 * a * b - inter)


def critical(a, b, thr):
    """Angle at which the ceiling falls to ``thr``, by bisection."""
    lo, hi = 0.0, 90.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if ceiling(a, b, mid) > thr else (lo, mid)
    return lo


def median_dims(info_path):
    with open(info_path, 'rb') as f:
        data_list = pickle.load(f)['data_list']
    dims = {c: [] for c in CLASSES}
    for sample in data_list:
        for inst in sample.get('instances', []):
            label = inst['bbox_label_3d']
            if label >= 0:
                dims[CLASSES[label]].append(inst['bbox_3d'][3:6])
    return {c: (np.median(np.asarray(v), axis=0) if v else None)
            for c, v in dims.items()}


def geometry_table(dims):
    print('IoU ceiling for a 7-DoF box against a target tilted by theta.\n')
    print('Pitch turns the box in the (length, height) plane and roll in the')
    print('(width, height) plane, so the aspect ratio of that plane is what')
    print('decides which angle hurts -- a long flat hull is punished by pitch')
    print('and barely touched by roll, a masted sailboat the reverse.\n')
    head = (f'{"class":9s} {"dx":>6s} {"dy":>6s} {"dz":>6s} {"thr":>5s} | ' +
            ' '.join(f'{"p" + str(int(a)):>6s}' for a in ANGLES) +
            f' {"crit":>6s} | ' +
            ' '.join(f'{"r" + str(int(a)):>6s}' for a in ANGLES) +
            f' {"crit":>6s}')
    print(head)
    print('-' * len(head))
    for c in CLASSES:
        d = dims.get(c)
        if d is None:
            continue
        dx, dy, dz = d
        thr = THRESHOLDS[c]
        pitch = [ceiling(dx, dz, a) for a in ANGLES]
        roll = [ceiling(dy, dz, a) for a in ANGLES]
        print(f'{c:9s} {dx:6.2f} {dy:6.2f} {dz:6.2f} {thr:5.2f} | ' +
              ' '.join(f'{v:6.3f}' for v in pitch) +
              f' {critical(dx, dz, thr):5.1f}d | ' +
              ' '.join(f'{v:6.3f}' for v in roll) +
              f' {critical(dy, dz, thr):5.1f}d')
    print('\n`p2` is the ceiling at 2 deg of pitch, `crit` the angle at which')
    print('the ceiling reaches the class threshold. `90.0d` means the ceiling')
    print('never falls that far: the box is close enough to square in that')
    print('plane that no rotation can push it out on its own.')


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, 'tensor'):
        x = x.tensor
    return x.detach().cpu().numpy()


def headroom_table(preds_path, dims, score_thr):
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir,
                                    'projects', 'Maritime3D'))
    from maritime3d.maritime_metric import _iou  # noqa: E402

    with open(preds_path, 'rb') as f:
        results = pickle.load(f)

    best = {c: [] for c in CLASSES}
    for sample in results:
        pred, gt = sample['pred_instances_3d'], sample['eval_ann_info']
        boxes = _to_numpy(pred['bboxes_3d'])
        labels = _to_numpy(pred['labels_3d'])
        scores = _to_numpy(pred['scores_3d'])
        gt_boxes = _to_numpy(gt['gt_bboxes_3d'])
        gt_labels = _to_numpy(gt['gt_labels_3d'])
        for ci, c in enumerate(CLASSES):
            p = boxes[(labels == ci) & (scores > score_thr)]
            g = gt_boxes[gt_labels == ci]
            if len(p) and len(g):
                # each ground-truth box keeps its best-overlapping detection
                best[c].extend(_iou(p, g, '3d').max(axis=0).tolist())

    print('\n\nShare of ground truth that currently clears the threshold and')
    print('would stop clearing it if the target were tilted.\n')
    head = (f'{"class":9s} {"GT":>6s} {"pass":>6s} | ' +
            ' '.join(f'{"pitch" + str(int(a)):>8s}' for a in ANGLES) + ' | ' +
            ' '.join(f'{"roll" + str(int(a)):>8s}' for a in ANGLES))
    print(head)
    print('-' * len(head))
    for c in CLASSES:
        v = np.asarray(best[c])
        d = dims.get(c)
        if not len(v) or d is None:
            continue
        dx, dy, dz = d
        thr = THRESHOLDS[c]
        passing = v >= thr
        n = max(int(passing.sum()), 1)
        cells = []
        for plane in ((dx, dz), (dy, dz)):
            cells.append([
                float(((v >= thr) & (v * ceiling(*plane, a) < thr)).sum()) / n
                for a in ANGLES
            ])
        print(f'{c:9s} {len(v):6d} {int(passing.sum()):6d} | ' +
              ' '.join(f'{x * 100:7.1f}%' for x in cells[0]) + ' | ' +
              ' '.join(f'{x * 100:7.1f}%' for x in cells[1]))
    print('\nThe ceiling is applied multiplicatively to the measured IoU,')
    print('which assumes the tilt error is independent of the centre and')
    print('extent error already present. That is an approximation, and the')
    print('reason to')
    print('read these as an order of magnitude rather than as three digits.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--infos', default='data/maritime/maritime_infos_train.pkl')
    p.add_argument('--preds', default=None,
                   help='DumpResults pkl; enables the headroom table')
    p.add_argument('--score-thr', type=float, default=0.1)
    args = p.parse_args()

    dims = median_dims(args.infos)
    geometry_table(dims)
    if args.preds:
        headroom_table(args.preds, dims, args.score_thr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
