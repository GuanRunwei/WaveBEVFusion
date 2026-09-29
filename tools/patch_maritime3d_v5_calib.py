#!/usr/bin/env python
"""Replace the placeholder camera calibration in the v5 pkls with the real
per-sequence calibration from dataset/<seq>/calibration/.

The v4/v5 info pkls shipped every frame with a placeholder images entry:
cam2img = [[1408, 0, 1024, 0], ...] (f = width/2, principal point = image
centre) and an identity lidar2cam.  Under that calibration no GT centre
projects into the stereo images, which kills every camera / fusion branch
and any projection-based visualisation.  The real calibration ships in
dataset/<seq>/calibration/{intrinsics,extrinsics}.json (seqs 01-03 inside
calibration.zip -- extracted to dataset/<seq>/calibration/ if missing).

For each stereo camera this script writes, per frame:
  cam2img   4x4 K from the real focal length / principal point (distortion
            is NOT folded in -- same convention as nuScenes; rectify first
            if a camera model needs it);
  lidar2cam 4x4 rigid transform computed from the extrinsics JSON;
  lidar2img cam2img @ lidar2cam.
img_path/height/width are untouched, and every annotation field stays
verbatim -- only the three calibration matrices are replaced.

The extrinsics JSON does not document its conventions (quaternion order
wxyz vs xyzw; sensor-in-body vs body-in-sensor), so the script scores all
four combinations by the fraction of GT box centres that project into the
stereo-left image on sequence 00 and keeps the best; the convention and its
score are printed and recorded in each pkl's metainfo.

Every pkl is rewritten atomically (tmp file + os.replace).
"""
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np


def quat_mat(q, order):
    """unit quaternion -> 3x3 rotation matrix; q given as (w,x,y,z) or
    (x,y,z,w) per `order`."""
    if order == 'wxyz':
        w, x, y, z = (float(v) for v in q)
    else:
        x, y, z, w = (float(v) for v in q)
    n = w*w + x*x + y*y + z*z
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

ROOT = Path(__file__).resolve().parents[1]
DET_DIR = ROOT / 'dataset/new_annotations/detection'
TRK_DIR = ROOT / 'dataset/new_annotations/tracking'
PKLS = [
    DET_DIR / 'maritime_nuscenes_infos_train_10dof_v5.pkl',
    DET_DIR / 'maritime_nuscenes_infos_val_10dof_v5.pkl',
    DET_DIR / 'maritime_nuscenes_infos_test_10dof_v5.pkl',
    TRK_DIR / 'maritime_nuscenes_infos_trainval_10dof_track_v5.pkl',
    TRK_DIR / 'maritime_nuscenes_infos_test_10dof_track_v5.pkl',
]
CAMS = [('CAM_FRONT_LEFT', 'stereo_left', 'left_images'),
        ('CAM_FRONT_RIGHT', 'stereo_right', 'right_images')]
SEQS = ['00', '01', '02', '03']


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def calib_dir(seq):
    d = ROOT / f'dataset/{seq}/calibration'
    if not d.is_dir():
        z = ROOT / f'dataset/{seq}/calibration.zip'
        if z.is_file():
            import zipfile
            with zipfile.ZipFile(z) as zf:
                zf.extractall(ROOT / f'dataset/{seq}')
    if not d.is_dir():
        raise FileNotFoundError(f'no calibration for seq {seq}')
    return d


def make_T(entry, order, direction):
    """4x4 transform from an extrinsics entry; direction 's2b' maps
    sensor->body, 'b2s' maps body->sensor (its inverse)."""
    R = quat_mat(entry['quaternion'], order)
    t = np.asarray(entry['translation'], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T if direction == 's2b' else np.linalg.inv(T)


def K_of(intr):
    f, cx, cy = intr['focal_length'], intr['cc_x'], intr['cc_y']
    return np.array([[f, 0, cx, 0],
                     [0, f, cy, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float64)


def proj_rate(frames, lidar2cam_by_seq, K_by_seq, sample=1500):
    """fraction of GT centres that land inside the stereo-left image."""
    hit, tot = 0, 0
    rng = np.random.RandomState(0)
    for s in rng.choice(len(frames), size=min(sample, len(frames)),
                        replace=False):
        f = frames[s]
        seq = f['lidar_path'].split('/')[0]
        K = K_by_seq['CAM_FRONT_LEFT'][seq]
        E = lidar2cam_by_seq['CAM_FRONT_LEFT'][seq]
        inst = f.get('cam_instances', {}).get('CAM_FRONT_LEFT') or \
            f.get('instances')
        if not inst:
            continue
        P = (K @ E)[:3]
        for it in inst:
            b = it['bbox_3d']
            c = P @ np.array([b[0], b[1], b[2], 1.0])
            if c[2] <= 0.1:
                continue
            u, v = c[0] / c[2], c[1] / c[2]
            tot += 1
            if 0 <= u < 2048 and 0 <= v < 1080:
                hit += 1
    return hit / max(1, tot), tot


def main():
    intr, extr = {}, {}
    for seq in SEQS:
        d = calib_dir(seq)
        intr[seq] = json.load(open(d / 'intrinsics.json'))
        extr[seq] = json.load(open(d / 'extrinsics.json'))

    # ---- pick conventions on seq00 by GT projection rate
    sample = load_pkl(DET_DIR / 'maritime_nuscenes_infos_val_10dof_v5.pkl')
    s00 = [f for f in sample['data_list']
           if f['lidar_path'].split('/')[0] == '00']
    best = None
    for order in ('wxyz', 'xyzw'):
        for direction in ('s2b', 'b2s'):
            E = {'CAM_FRONT_LEFT': {}, 'CAM_FRONT_RIGHT': {}}
            K = {'CAM_FRONT_LEFT': {}, 'CAM_FRONT_RIGHT': {}}
            for seq in SEQS:
                T_body_lidar = make_T(extr[seq]['lidar_front'], order,
                                      direction)
                for cam, sensor, _ in CAMS:
                    T_body_cam = make_T(extr[seq][sensor], order, direction)
                    E[cam][seq] = T_body_cam @ np.linalg.inv(T_body_lidar)
                    K[cam][seq] = K_of(intr[seq][sensor])
            rate, tot = proj_rate(s00, E, K)
            print(f'convention {order}/{direction}: projection rate '
                  f'{rate:.1%} of {tot} GT centres')
            if best is None or rate > best[0]:
                best = (rate, order, direction, E, K)
    rate, order, direction, E, K = best
    print(f'selected: quaternion={order}, T={direction} '
          f'(seq00 projection rate {rate:.1%})')
    if rate < 0.05:
        sys.exit('selected convention still projects <5% of GT -- '
                 'extrinsics frame assumptions need re-examination')

    # ---- rewrite pkls atomically
    for path in PKLS:
        d = load_pkl(path)
        frames = d['data_list']
        pre, tot = proj_rate(frames, E, K, sample=1200)
        n_changed = 0
        for f in frames:
            seq = f['lidar_path'].split('/')[0]
            imgs = f.get('images') or {}
            for cam, sensor, _ in CAMS:
                ent = imgs.get(cam)
                if ent is None:
                    continue
                ent['cam2img'] = K[cam][seq].tolist()
                ent['lidar2cam'] = E[cam][seq].tolist()
                ent['lidar2img'] = (K[cam][seq] @ E[cam][seq]).tolist()
                n_changed += 1
            f['images'] = imgs
        d['metainfo']['calibration'] = (
            f'real per-sequence intrinsics/extrinsics; quaternion={order}, '
            f'T={direction}; patched by tools/patch_maritime3d_v5_calib.py')
        tmp = path.with_suffix('.pkl.tmp')
        with open(tmp, 'wb') as fh:
            pickle.dump(d, fh)
        os.replace(tmp, path)
        post, tot2 = proj_rate(frames, E, K, sample=1200)
        print(f'{path.name}: {n_changed} image entries patched, '
              f'projection rate {pre:.1%} -> {post:.1%}')


if __name__ == '__main__':
    main()
