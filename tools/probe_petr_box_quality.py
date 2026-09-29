"""Why the camera-only baseline scores 0 AP: measure where its boxes land.

AP40 at IoU 0.5 is a pass/fail gate. When a method scores exactly 0 on every
class the number says only "no box cleared the gate" -- it does not say whether
the model learned nothing or learned a lot and missed the gate by one metre.
For a benchmark that distinction is the whole point, so this measures the
localisation error directly.

For every ground-truth object inside the camera sector it takes the highest-
scoring prediction of the same class within ``--assoc`` metres in BEV and
reports the error decomposition. The interesting split is lateral (azimuth,
which a camera measures directly from pixel position) against radial (depth,
which a forward-facing pair with a short baseline has to infer). If radial
error dominates and grows with range, 0 AP is the rig talking, not the method.

    python tools/probe_petr_box_quality.py --epoch 40
"""
import argparse
import os
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner, load_checkpoint

sys.path.insert(0, os.getcwd())
from mmdet3d.registry import MODELS  # noqa: E402

CFG = 'projects/Maritime3D/configs/petr_maritime-3d-4class.py'
AZ = (-35.0, 24.0)
BANDS = [(0, 50), (50, 100), (100, 160)]


def in_sector(boxes):
    if len(boxes) == 0:
        return np.zeros(0, dtype=bool)
    az = np.degrees(np.arctan2(boxes[:, 1], boxes[:, 0]))
    return (az >= AZ[0]) & (az <= AZ[1])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epoch', type=int, default=40)
    p.add_argument('--assoc', type=float, default=60.0,
                   help='BEV radius for calling a prediction the same object')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()

    cfg = Config.fromfile(CFG)
    init_default_scope('mmdet3d')
    loader = Runner.build_dataloader(cfg.val_dataloader)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, f'work_dirs/petr/epoch_{args.epoch}.pth',
                    map_location='cpu')
    model.to(args.device).eval()

    rows = []          # gt_range, radial, lateral, matched, gt_az, pred_az
    n_gt = n_pred = 0
    with torch.no_grad():
        for data in loader:
            for s in model.test_step(data):
                s = s.to_dict()
                pb = s['pred_instances_3d']['bboxes_3d'].tensor.cpu().numpy()
                ps = s['pred_instances_3d']['scores_3d'].cpu().numpy()
                pl = s['pred_instances_3d']['labels_3d'].cpu().numpy()
                gt = s['eval_ann_info']['gt_bboxes_3d']
                gb = gt if isinstance(gt, np.ndarray) else gt.tensor.numpy()
                gb = np.asarray(gb, dtype=np.float32).reshape(-1, 7)
                gl = np.asarray(s['eval_ann_info']['gt_labels_3d']).reshape(-1)

                km, kg = in_sector(pb), in_sector(gb)
                pb, ps, pl = pb[km], ps[km], pl[km]
                gb, gl = gb[kg], gl[kg]
                n_gt += len(gb)
                n_pred += len(pb)
                for g, lab in zip(gb, gl):
                    r = float(np.hypot(g[0], g[1]))
                    gaz = float(np.degrees(np.arctan2(g[1], g[0])))
                    cand = np.where(pl == lab)[0]
                    if len(cand) == 0:
                        rows.append((r, np.nan, np.nan, 0, gaz, np.nan))
                        continue
                    d = np.hypot(pb[cand, 0] - g[0], pb[cand, 1] - g[1])
                    near = cand[d <= args.assoc]
                    if len(near) == 0:
                        rows.append((r, np.nan, np.nan, 0, gaz, np.nan))
                        continue
                    b = near[np.argmax(ps[near])]
                    # decompose the centre error along the line of sight
                    u = np.array([g[0], g[1]]) / max(r, 1e-6)
                    e = np.array([pb[b, 0] - g[0], pb[b, 1] - g[1]])
                    rad = float(abs(e @ u))
                    lat = float(abs(e[0] * -u[1] + e[1] * u[0]))
                    paz = float(np.degrees(np.arctan2(pb[b, 1], pb[b, 0])))
                    rows.append((r, rad, lat, 1, gaz, paz))

    a = np.array(rows, dtype=float)
    print(f'\ncamera-sector val: {n_gt} gt objects, {n_pred} predictions, '
          f'{int(a[:, 3].sum())} associated within {args.assoc:.0f} m\n')
    print(f'{"range":>10s} {"n":>5s} {"assoc%":>7s} '
          f'{"radial(m)":>10s} {"lateral(m)":>11s} {"rad/range":>10s}')
    for lo, hi in BANDS:
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if not m.any():
            continue
        sub = a[m]
        ok = sub[sub[:, 3] == 1]
        rad = np.median(ok[:, 1]) if len(ok) else np.nan
        lat = np.median(ok[:, 2]) if len(ok) else np.nan
        mid = (lo + hi) / 2
        print(f'{f"{lo}-{hi}m":>10s} {len(sub):5d} '
              f'{100 * sub[:, 3].mean():6.1f}% {rad:10.2f} {lat:11.2f} '
              f'{rad / mid:9.1%}')
    ok = a[a[:, 3] == 1]
    print(f'{"all":>10s} {len(a):5d} {100 * a[:, 3].mean():6.1f}% '
          f'{np.median(ok[:, 1]):10.2f} {np.median(ok[:, 2]):11.2f}')

    # Azimuth is the one quantity a camera measures directly -- it is pixel
    # column, no depth inference required. If the predicted azimuth does not
    # track the ground-truth azimuth, the model is not reading the image, and
    # no amount of range accuracy will save the IoU.
    gaz, paz = ok[:, 4], ok[:, 5]
    print(f'\nazimuth (deg), {len(ok)} associated objects')
    print(f'  gt    spread: {gaz.std():6.2f}  range [{gaz.min():.1f}, '
          f'{gaz.max():.1f}]')
    print(f'  pred  spread: {paz.std():6.2f}  range [{paz.min():.1f}, '
          f'{paz.max():.1f}]')
    print(f'  corr(gt, pred): {np.corrcoef(gaz, paz)[0, 1]:+.3f}   '
          f'median |error|: {np.median(np.abs(paz - gaz)):.2f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
