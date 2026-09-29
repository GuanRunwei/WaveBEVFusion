#!/usr/bin/env python
"""Write compact (N, 4) point clouds for every frame referenced by the tables.

The raw Pohang ``lidar_front/points/*.bin`` files are fixed 131072x7 float32
buffers (3.7 MB). Only x, y, z, intensity are populated, and ~75% of the rows
are all-zero padding. This script drops the padding rows and the three empty
columns and writes ``dataset/points4/<seq>/<timestamp>.bin`` as float32
(x, y, z, intensity). That is ~6x less I/O per frame, and the whole set (~50 GB)
fits in the page cache.

Usage:
    python tools/compact_maritime_points.py --workers 64
"""
import argparse
import json
import os
from multiprocessing import Pool

import numpy as np

ROOT = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset'))
RAW_PREFIX = 'Pohang Canal Dataset/'


def lidar_files(table_dir):
    """Relative raw paths ('00/lidar_front/points/<ts>.bin') in the tables."""
    with open(os.path.join(table_dir, 'sample_data.json')) as f:
        sd = json.load(f)
    out = set()
    for s in sd:
        fn = s['filename']
        if '/lidar_front/' in fn:
            out.add(fn[len(RAW_PREFIX):] if fn.startswith(RAW_PREFIX) else fn)
    return sorted(out)


def compact_path(raw_rel):
    """'00/lidar_front/points/<ts>.bin' -> 'points4/00/<ts>.bin'."""
    seq = raw_rel.split('/')[0]
    return os.path.join('points4', seq, os.path.basename(raw_rel))


def convert(raw_rel):
    dst = os.path.join(ROOT, compact_path(raw_rel))
    if os.path.exists(dst):
        return raw_rel, -1
    a = np.fromfile(os.path.join(ROOT, raw_rel), dtype=np.float32)
    a = a.reshape(-1, 7)
    keep = np.abs(a[:, :3]).sum(1) > 1e-6
    pts = np.ascontiguousarray(a[keep, :4])
    tmp = dst + '.tmp'
    pts.tofile(tmp)
    os.replace(tmp, dst)
    return raw_rel, len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--tables',
        nargs='+',
        default=[
            os.path.join(ROOT, 'splits/tables/detection', f'v1.0-{s}')
            for s in ('train', 'val', 'test')
        ],
        help='table dirs whose sample_data.json together list every frame '
        '(v1.0-full/sample_data.json is truncated on disk, so the default '
        'is the union of the three detection splits)')
    ap.add_argument('--workers', type=int, default=64)
    args = ap.parse_args()

    files = sorted(set().union(*(lidar_files(t) for t in args.tables)))
    for seq in sorted({f.split('/')[0] for f in files}):
        os.makedirs(os.path.join(ROOT, 'points4', seq), exist_ok=True)
    print(f'{len(files)} frames to compact', flush=True)
    n_done, n_skip, counts = 0, 0, []
    with Pool(args.workers) as pool:
        for i, (_, n) in enumerate(
                pool.imap_unordered(convert, files, chunksize=16)):
            if n < 0:
                n_skip += 1
            else:
                n_done += 1
                counts.append(n)
            if (i + 1) % 5000 == 0:
                print(f'  {i + 1}/{len(files)}', flush=True)
    counts = np.asarray(counts)
    print(f'done: wrote {n_done}, skipped {n_skip} existing', flush=True)
    if len(counts):
        print(f'points/frame: min {counts.min()} median '
              f'{int(np.median(counts))} max {counts.max()}', flush=True)


if __name__ == '__main__':
    main()
