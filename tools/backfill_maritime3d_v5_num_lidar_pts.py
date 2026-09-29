#!/usr/bin/env python
"""Backfill the real per-box num_lidar_pts from the v5 tracking tables into
the v5 info pkls.

The new_annotations pkls ship num_lidar_pts=100 for every instance (a
placeholder -- see VALIDATION_REPORT_v5.md), which silently disables any
difficulty / point-support filtering.  The nuScenes-style tracking tables
(tracking_v5/v1.0-{trainval,test}/sample_annotation.json) carry the REAL
per-annotation counts inherited from the source labels, and the audit
verified per-sample instance counts and size multisets match the pkls
exactly, so the tables are the authoritative source.

Matching is geometric per sample token: the pkl instance dicts carry no
instance token, so each pkl box [x, y, z, dx, dy, dz, ...] is matched to the
table row with the same size (dx, dy, dz within 1e-3) and the nearest (x, y)
centre (table translations are box centres; pkl z is the bottom, so z is not
compared).  This also disambiguates the 131 source rows where one instance
token appears twice in a frame (distinct geometry).

145 table rows have num_lidar_pts=null (inherited source defect); those keep
None in the pkl rather than inventing a value.

Every pkl is rewritten atomically (tmp + os.replace).  Only the
num_lidar_pts field of each instance is touched; bbox/labels/calibration
stay verbatim.
"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET = ROOT / 'dataset/new_annotations/detection'
TRK = ROOT / 'dataset/new_annotations/tracking'
PKLS = {
    'trainval': [
        DET / 'maritime_nuscenes_infos_train_10dof_v5.pkl',
        DET / 'maritime_nuscenes_infos_val_10dof_v5.pkl',
        TRK / 'maritime_nuscenes_infos_trainval_10dof_track_v5.pkl',
    ],
    'test': [
        DET / 'maritime_nuscenes_infos_test_10dof_v5.pkl',
        TRK / 'maritime_nuscenes_infos_test_10dof_track_v5.pkl',
    ],
}
SZ_TOL = 1e-3


def load_table_rows(split):
    """sample_token -> list of (size3, xy, num_lidar_pts) rows."""
    anns = json.load(
        open(TRK / 'tracking_v5' / f'v1.0-{split}' / 'sample_annotation.json'))
    by_sample = {}
    for a in anns:
        by_sample.setdefault(a['sample_token'], []).append(
            (np.asarray(a['size'], dtype=np.float64),
             np.asarray(a['translation'][:2], dtype=np.float64),
             a.get('num_lidar_pts')))
    return by_sample


def match_frame(instances, rows):
    """Greedy size-then-nearest-centre assignment; returns per-instance row
    index or None (unmatched)."""
    remaining = list(range(len(rows)))
    out = [None] * len(instances)
    # pass 1: exact size match, nearest centre (most-constrained boxes first)
    order = sorted(range(len(instances)),
                   key=lambda i: -float(np.prod(instances[i]['bbox_3d'][3:6])))
    for i in order:
        b = instances[i]['bbox_3d']
        best, best_d = None, None
        for r in remaining:
            if np.all(np.abs(rows[r][0] - b[3:6]) <= SZ_TOL):
                d = float(np.sum((rows[r][1] - b[:2]) ** 2))
                if best_d is None or d < best_d:
                    best, best_d = r, d
        if best is not None:
            out[i] = best
            remaining.remove(best)
    return out, remaining


def main():
    tables = {sp: load_table_rows(sp) for sp in PKLS}
    for split, paths in PKLS.items():
        by_sample = tables[split]
        for path in paths:
            with open(path, 'rb') as f:
                d = pickle.load(f)
            frames = d['data_list']
            n_inst = n_fill = n_none = n_unmatched = 0
            vals = []
            for fr in frames:
                rows = by_sample.get(fr['token'])
                insts = fr.get('instances', [])
                n_inst += len(insts)
                if not insts:
                    continue
                if rows is None or len(rows) != len(insts):
                    n_unmatched += len(insts)
                    continue
                assign, leftover = match_frame(insts, rows)
                if leftover:
                    n_unmatched += len(leftover)
                    continue
                for it, r in zip(insts, assign):
                    v = rows[r][2]
                    if v is None:
                        it['num_lidar_pts'] = None
                        n_none += 1
                    else:
                        it['num_lidar_pts'] = int(v)
                        vals.append(v)
                        n_fill += 1
            tmp = path.with_suffix('.pkl.tmp')
            with open(tmp, 'wb') as fh:
                pickle.dump(d, fh)
            os.replace(tmp, path)
            qs = np.percentile(vals, [0, 25, 50, 75, 100]).astype(int) \
                if vals else [0] * 5
            print(f'{path.name}: {n_inst} instances, {n_fill} backfilled '
                  f'({n_none} None, {n_unmatched} unmatched), '
                  f'pts quartiles {list(qs)}')


if __name__ == '__main__':
    main()
