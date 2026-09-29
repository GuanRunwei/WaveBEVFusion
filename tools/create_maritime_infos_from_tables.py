#!/usr/bin/env python
"""Build mmdet3d info pkls from the nuScenes-style split tables.

Input:  dataset/splits/tables/<protocol>/v1.0-<split>/  (make_splits.py output)
Output: dataset/infos/<protocol>/maritime_infos_<split>.pkl

Conventions (all verified on the data, see projects/Maritime3D/figs/ and
projects/Maritime3D/BENCHMARK_MODEL_DESIGN.md §2):

* Boxes: the tables' ``translation`` is the geometric centre in the LiDAR
  frame and ``size`` is (w, l, h). Yaw/pitch/roll come from the wxyz
  quaternion. We store
  ``bbox_3d = [x, y, z_bottom, l, w, h, yaw, pitch, roll]``.
  MaritimeDataset builds boxes with origin (0.5, 0.5, 0) and keeps the first
  ``box_3d_dof`` (7 or 9) columns. Points counted in these boxes equal the
  tables' ``num_lidar_pts``, and ``--verify`` re-checks that.
* Calibration: ``dataset/<seq>/calibration/extrinsics.json`` quaternions are
  **xyzw** (ROS order) sensor->body poses, so
  lidar2cam = inv(T_body_cam) @ T_body_lidar. (tools/
  patch_maritime3d_v5_calib.py picked wxyz, which mirrors every projection.)
  cam2img is the 3x3 pinhole K (lidar2img stays 4x4). Distortion coefficients are stored alongside;
  images are NOT rectified.
* Points: ``lidar_path`` points at the compact (N, 4) float32 files written
  by tools/compact_maritime_points.py (``points4/<seq>/<ts>.bin``), so
  configs load with load_dim=use_dim=4. ``lidar_path_raw`` keeps the
  original 131072x7 buffer.

Extra per-frame fields (used by the maritime-specific parts of the model and
metric): seq, daynight, scene_name, radar (nearest preceding sweep, seqs
01-03), sea_plane (robust plane z = a*x + b*y + c fitted to GT box bottoms,
None if under-determined). Per instance: in_fov (centre projects into
either camera), num_lidar_pts, instance_token.

Usage:
    python tools/create_maritime_infos_from_tables.py --verify 300
"""
import argparse
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'))
RAW_PREFIX = 'Pohang Canal Dataset/'
CLASSES = ('boat', 'buoy', 'sailboat', 'ship', 'yacht')
CAMS = (('CAM_FRONT_LEFT', 'stereo_left', 'left_images'),
        ('CAM_FRONT_RIGHT', 'stereo_right', 'right_images'))
IMG_H, IMG_W = 1080, 2048


# --------------------------------------------------------------- geometry
def quat_xyzw_to_mat(q):
    x, y, z, w = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def pose(entry):
    """sensor->body 4x4 from an extrinsics.json entry (xyzw quaternion)."""
    T = np.eye(4)
    T[:3, :3] = quat_xyzw_to_mat(entry['quaternion'])
    T[:3, 3] = entry['translation']
    return T


def euler_from_wxyz(q):
    """(yaw, pitch, roll), ZYX convention, from a wxyz quaternion."""
    w, x, y, z = (float(v) for v in q)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    return yaw, pitch, roll


def load_calib(seq):
    d = os.path.join(ROOT, seq, 'calibration')
    extr = json.load(open(os.path.join(d, 'extrinsics.json')))
    intr = json.load(open(os.path.join(d, 'intrinsics.json')))
    T_body_lidar = pose(extr['lidar_front'])
    out = {}
    for cam, sensor, _ in CAMS:
        ic = intr[sensor]
        K = np.eye(4)
        K[0, 0] = K[1, 1] = ic['focal_length']
        K[0, 2], K[1, 2] = ic['cc_x'], ic['cc_y']
        lidar2cam = np.linalg.inv(pose(extr[sensor])) @ T_body_lidar
        out[cam] = dict(
            cam2img=K,
            lidar2cam=lidar2cam,
            lidar2img=K @ lidar2cam,
            distortion=list(ic['distortion_coefficients']),
            height=int(ic['image_height']),
            width=int(ic['image_width']))
    return out


def fit_sea_plane(bottoms, min_boxes=3, min_spread=3.0, n_iter=3, clip=1.5):
    """Robust plane z = a*x + b*y + c through GT box bottom centres.

    Returns (plane[3] or None, n_inliers). Under-determined frames (fewer
    than ``min_boxes`` boxes, or nearly collinear in xy) get None.
    """
    if len(bottoms) < min_boxes:
        return None, 0
    xy = bottoms[:, :2] - bottoms[:, :2].mean(0)
    if np.linalg.svd(xy, compute_uv=False)[-1] / np.sqrt(
            len(bottoms)) < min_spread:
        return None, 0
    keep = np.ones(len(bottoms), bool)
    coef = None
    for _ in range(n_iter):
        if keep.sum() < min_boxes:
            break
        X = np.c_[bottoms[keep, :2], np.ones(keep.sum())]
        coef = np.linalg.lstsq(X, bottoms[keep, 2], rcond=None)[0]
        res = np.abs(np.c_[bottoms[:, :2], np.ones(len(bottoms))] @ coef -
                     bottoms[:, 2])
        new_keep = res < clip
        if (new_keep == keep).all():
            break
        keep = new_keep
    if coef is None or keep.sum() < min_boxes:
        return None, 0
    return [float(c) for c in coef], int(keep.sum())


# ------------------------------------------------------------ side tables
def load_frame_timestamps(path):
    """'<unix s.ns>\\t<frame idx>' lines -> {idx: ns}."""
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            sec, _, frac = p[0].partition('.')
            out[int(p[1])] = int(sec) * 10**9 + int(frac.ljust(9, '0')[:9])
    return out


class RadarIndex:
    """Nearest *preceding* radar sweep for a LiDAR timestamp (causal)."""

    def __init__(self, seq):
        p = os.path.join(ROOT, seq, 'radar', 'timestamp.txt')
        self.ok = os.path.exists(p)
        if not self.ok:
            return
        ts = load_frame_timestamps(p)
        self.idx = np.array(sorted(ts))
        self.ts = np.array([ts[i] for i in self.idx], dtype=np.int64)
        self.seq = seq

    def query(self, t_ns):
        if not self.ok:
            return None
        k = int(np.searchsorted(self.ts, t_ns, side='right')) - 1
        if k < 0:
            return None
        return dict(
            img_path=f'{self.seq}/radar/images/{self.idx[k]:06d}.png',
            frame_idx=int(self.idx[k]),
            timestamp=int(self.ts[k]),
            lidar_minus_radar_s=float((t_ns - self.ts[k]) * 1e-9))


# -------------------------------------------------------------- converter
def strip(fn):
    return fn[len(RAW_PREFIX):] if fn.startswith(RAW_PREFIX) else fn


def in_any_fov(center, cams):
    """True if a LiDAR-frame point projects into either camera image."""
    p = np.r_[center, 1.0]
    for c in cams.values():
        q = c['lidar2img'] @ p
        if q[2] > 0.5:
            u, v = q[0] / q[2], q[1] / q[2]
            if 0 <= u < c['width'] and 0 <= v < c['height']:
                return True
    return False


def convert_split(table_dir, split, calib, stereo_ts, radar):
    def J(name):
        with open(os.path.join(table_dir, name + '.json')) as f:
            return json.load(f)

    cats = {c['token']: c['name'] for c in J('category')}
    inst_cls = {i['token']: cats[i['category_token']] for i in J('instance')}
    scenes = {s['token']: s for s in J('scene')}
    samples = J('sample')
    files = defaultdict(dict)
    for s in J('sample_data'):
        fn = strip(s['filename'])
        if '/lidar_front/' in fn:
            files[s['sample_token']]['lidar'] = fn
        elif '/left_images/' in fn:
            files[s['sample_token']]['CAM_FRONT_LEFT'] = fn
        elif '/right_images/' in fn:
            files[s['sample_token']]['CAM_FRONT_RIGHT'] = fn
    anns = defaultdict(list)
    for a in J('sample_annotation'):
        anns[a['sample_token']].append(a)

    # scene order, then time order inside each scene
    samples.sort(key=lambda s: (scenes[s['scene_token']]['name'],
                                s['timestamp']))
    data_list, stats = [], defaultdict(int)
    for idx, s in enumerate(samples):
        tok = s['token']
        fs = files[tok]
        seq = fs['lidar'].split('/')[0]
        scene = scenes[s['scene_token']]
        m = re.search(r'\((day|night)\)', scene.get('description', ''))
        cams = calib[seq]
        lidar_raw = fs['lidar']
        info = dict(
            token=tok,
            sample_idx=idx,
            timestamp=int(s['timestamp']),
            scene_token=s['scene_token'],
            scene_name=scene['name'],
            seq=seq,
            daynight=m.group(1) if m else 'unknown',
            lidar_points=dict(
                lidar_path=os.path.join('points4', seq,
                                        os.path.basename(lidar_raw)),
                lidar_path_raw=lidar_raw,
                num_pts_feats=4),
            images={},
            radar=radar[seq].query(int(s['timestamp'])),
        )
        for cam, _, _ in CAMS:
            c = cams[cam]
            img = fs[cam]
            fidx = int(os.path.splitext(os.path.basename(img))[0])
            t_img = stereo_ts[seq].get(fidx)
            info['images'][cam] = dict(
                img_path=img,
                height=c['height'],
                width=c['width'],
                # 3x3, the mmdet3d/nuScenes convention BEVFusion's loader expects
                cam2img=c['cam2img'][:3, :3].tolist(),
                lidar2cam=c['lidar2cam'].tolist(),
                lidar2img=c['lidar2img'].tolist(),
                distortion=c['distortion'],
                timestamp=t_img,
                cam_sync_dt=None if t_img is None else
                float((t_img - int(s['timestamp'])) * 1e-9))
        instances, bottoms = [], []
        for a in anns.get(tok, []):
            name = inst_cls[a['instance_token']]
            w, l, h = (float(v) for v in a['size'])
            x, y, z = (float(v) for v in a['translation'])
            yaw, pitch, roll = euler_from_wxyz(a['rotation'])
            label = CLASSES.index(name)
            n_pts = a.get('num_lidar_pts')
            instances.append(
                dict(
                    bbox_3d=[x, y, z - h / 2, l, w, h, yaw, pitch, roll],
                    bbox_label_3d=label,
                    bbox_label=label,
                    num_lidar_pts=-1 if n_pts is None else int(n_pts),
                    in_fov=bool(in_any_fov(np.array([x, y, z]), cams)),
                    instance_token=a['instance_token']))
            bottoms.append([x, y, z - h / 2])
            stats[f'box_{name}'] += 1
        plane, n_sup = fit_sea_plane(np.asarray(bottoms).reshape(-1, 3))
        info['instances'] = instances
        info['sea_plane'] = plane
        info['sea_plane_support'] = n_sup
        stats['frames'] += 1
        stats['frames_empty'] += int(not instances)
        stats['frames_with_plane'] += int(plane is not None)
        stats['frames_with_radar'] += int(info['radar'] is not None)
        data_list.append(info)
    return data_list, dict(stats)


# ------------------------------------------------------------ verification
def load_points(info):
    p = os.path.join(ROOT, info['lidar_points']['lidar_path'])
    if os.path.exists(p):
        return np.fromfile(p, dtype=np.float32).reshape(-1, 4)
    a = np.fromfile(
        os.path.join(ROOT, info['lidar_points']['lidar_path_raw']),
        dtype=np.float32).reshape(-1, 7)
    return a[np.abs(a[:, :3]).sum(1) > 1e-6, :4]


def count_in_box(pts, b):
    x, y, zb, l, w, h, yaw = b[:7]
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    c, s = np.cos(-yaw), np.sin(-yaw)
    lx, ly = dx * c - dy * s, dx * s + dy * c
    dz = pts[:, 2] - zb
    return int(((np.abs(lx) < l / 2) & (np.abs(ly) < w / 2) & (dz > 0) &
                (dz < h)).sum())


def verify(data_list, n, seed=0):
    """Points-in-box must reproduce the tables' num_lidar_pts."""
    rng = np.random.RandomState(seed)
    cand = [i for i, d in enumerate(data_list) if d['instances']]
    pick = rng.choice(cand, size=min(n, len(cand)), replace=False)
    n_box = n_exact = n_close = 0
    for i in pick:
        d = data_list[i]
        pts = load_points(d)
        for inst in d['instances']:
            if inst['num_lidar_pts'] < 0:
                continue
            got = count_in_box(pts, inst['bbox_3d'])
            ref = inst['num_lidar_pts']
            n_box += 1
            n_exact += int(got == ref)
            n_close += int(abs(got - ref) <= max(2, 0.02 * ref))
    return n_box, n_exact, n_close


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', default='detection',
                    help='subdir of splits/tables (detection, ood_nightshift)')
    ap.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--out-dir', default=None)
    ap.add_argument('--verify', type=int, default=0,
                    help='frames per split to re-count points-in-box')
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(ROOT, 'infos', args.protocol)
    os.makedirs(out_dir, exist_ok=True)
    seqs = ('00', '01', '02', '03')
    calib = {q: load_calib(q) for q in seqs}
    stereo_ts = {
        q: load_frame_timestamps(
            os.path.join(ROOT, q, 'stereo', 'timestamp.txt'))
        for q in seqs
    }
    radar = {q: RadarIndex(q) for q in seqs}

    for split in args.splits:
        table_dir = os.path.join(ROOT, 'splits', 'tables', args.protocol,
                                 f'v1.0-{split}')
        data_list, stats = convert_split(table_dir, split, calib, stereo_ts,
                                         radar)
        metainfo = dict(
            dataset='maritime3d',
            info_version='2.0',
            protocol=args.protocol,
            split=split,
            classes=list(CLASSES),
            categories={c: i for i, c in enumerate(CLASSES)},
            tables=os.path.relpath(table_dir, ROOT),
            box_format='bbox_3d = [x, y, z_bottom, l, w, h, yaw, pitch, '
            'roll], LiDAR frame',
            calibration='extrinsics.json quaternions read as xyzw, '
            'sensor->body; lidar2cam = inv(T_body_cam) @ T_body_lidar; '
            'images not rectified (distortion stored per camera)',
            points='points4/<seq>/<ts>.bin, float32 (x, y, z, intensity)')
        out = os.path.join(out_dir, f'maritime_infos_{split}.pkl')
        tmp = out + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(dict(metainfo=metainfo, data_list=data_list), f)
        os.replace(tmp, out)
        n_fov = sum(i['in_fov'] for d in data_list for i in d['instances'])
        n_box = sum(len(d['instances']) for d in data_list)
        print(f'[{split}] {out}')
        print(f'  {stats}')
        print(f'  in-FOV boxes: {n_fov}/{n_box} '
              f'({n_fov / max(1, n_box):.1%})')
        if args.verify:
            nb, ne, nc = verify(data_list, args.verify)
            print(f'  verify: {nb} boxes, points-in-box == num_lidar_pts '
                  f'for {ne / max(1, nb):.2%} (within 2%: '
                  f'{nc / max(1, nb):.2%})')


if __name__ == '__main__':
    main()
