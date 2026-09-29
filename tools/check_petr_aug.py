"""Check that the camera-only augmentation keeps images and 3D boxes aligned.

Rotating the world while forgetting to rotate the cameras is silent: the loss
still goes down, the model just learns from mislabelled images. So reproject
every ground-truth corner through ``lidar2cam`` / ``cam2img`` before and after
``MaritimeGlobalRotScaleTransImage`` and assert the pixels do not move.

Usage:
    python tools/check_petr_aug.py [config]
"""
import argparse
import copy
import importlib
import sys

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope

DEFAULT_CFG = 'projects/Maritime3D/configs/petr_maritime-3d-4class.py'
# a corner behind the camera or far off-image reprojects to a meaningless
# coordinate; only pixels anywhere near the sensor are worth comparing
PIXEL_SANITY_LIMIT = 5000
TOLERANCE_PX = 1e-2


def project(results):
    """Project every GT corner into every camera. Returns (n_cam, n_pt, 2)."""
    corners = results['gt_bboxes_3d'].corners.numpy().reshape(-1, 3)
    homo = np.concatenate([corners, np.ones((len(corners), 1))], axis=1)
    out = []
    for view in range(len(results['lidar2cam'])):
        lidar2cam = np.asarray(results['lidar2cam'][view], dtype=np.float64)
        cam2img = np.eye(4)
        k = np.asarray(results['cam2img'][view], dtype=np.float64)
        cam2img[:k.shape[0], :k.shape[1]] = k
        proj = (cam2img @ lidar2cam @ homo.T).T
        out.append(proj[:, :2] / proj[:, 2:3])
    return np.stack(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', nargs='?', default=DEFAULT_CFG)
    parser.add_argument('--frames', type=int, default=40)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    for module in cfg.custom_imports['imports']:
        importlib.import_module(module)
    init_default_scope('mmdet3d')
    from mmdet3d.registry import DATASETS, TRANSFORMS

    aug_cfg = next(t for t in cfg.train_pipeline
                   if t['type'] == 'MaritimeGlobalRotScaleTransImage')
    stripped = [
        t for t in cfg.train_pipeline
        if t['type'] not in (aug_cfg['type'], 'Pack3DDetInputs')
    ]
    ds_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    ds_cfg['dataset']['pipeline'] = stripped
    dataset = DATASETS.build(ds_cfg).dataset

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    aug = TRANSFORMS.build(copy.deepcopy(aug_cfg))

    worst, checked = 0.0, 0
    for idx in range(len(dataset)):
        if checked >= args.frames:
            break
        results = dataset.pipeline(dataset.get_data_info(idx))
        if results is None or len(results['gt_bboxes_3d']) == 0:
            continue
        before = project(results)
        after = project(aug.transform(copy.deepcopy(results)))
        keep = (np.abs(before) < PIXEL_SANITY_LIMIT).all(-1)
        if not keep.any():
            continue
        worst = max(worst, float(np.abs(before - after)[keep].max()))
        checked += 1

    print(f'{aug}\nframes checked: {checked}\n'
          f'worst corner reprojection drift: {worst:.6f} px')
    if worst > TOLERANCE_PX:
        print(f'FAIL: drift exceeds {TOLERANCE_PX} px -- the augmentation '
              'moves the world without moving the cameras')
        return 1
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
