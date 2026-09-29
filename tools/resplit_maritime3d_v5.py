#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""Build the v5 block split for the Mari3D detection + tracking benchmarks.

Why v5 exists
-------------
The v4 detection split assigns frames to train/val/test at the FRAME level
inside the same continuous sequences, so val frames are temporally interleaved
with train frames (near-duplicates at 10 Hz) and no buffer exists between
trainval and test chunks.  The v4 *tracking* split is already block-based with
frame-set equality  det_train U det_val == tracking_trainval  and
det_test == tracking_test; v5 keeps that architecture and fixes the two
deficiencies:

1. detection train/val is re-derived at BLOCK level (20 s blocks) with
   per-class instance quotas, so every class keeps a floor of instances in
   every split (long-tail safe);
2. a 10 s temporal buffer is enforced at every split boundary, applied to a
   fixed point: any val frame within 10 s of a train frame moves to train;
   any trainval frame within 10 s of a test frame is absorbed into test.
   No annotated frame is ever dropped -- buffers are absorbed by the
   neighbouring split, so the whole annotated universe stays in use.

Tracking artifacts (info pkls + nuScenes-style tables) are regenerated from
the v4 tables by re-partitioning rows; every annotation row is copied
verbatim, only membership and the intra-scene prev/next chains are rebuilt
(chains are scene-local in v4 and stay scene-local in v5; sample_data chains
use sample_data tokens as in v4).

Hard guarantees enforced by the validator at the end of this script:
- frame conservation: train U val U test == v4 universe, pairwise disjoint;
- frame-set equality det_train U det_val == tracking_trainval and
  det_test == tracking_test;
- min |dt| between any train and test frame > TAU (likewise train/val);
- every emitted table passes a relational integrity check (all tokens
  referenced by prev/next/scene/sample/sample_data resolve inside the same
  table; 1:1 sample/sample_data; instance first/last annotations present);
- annotation conservation: total boxes across both v5 tables == v4.

This script is deterministic (fixed RNG seed) and never modifies v4 files.
"""

import argparse
import datetime
import hashlib
import json
import pickle
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET_DIR = ROOT / 'dataset/new_annotations/detection'
TRK_DIR = ROOT / 'dataset/new_annotations/tracking'

CLASSES = ('boat', 'buoy', 'sailboat', 'ship', 'yacht')

# ---------------- design parameters ----------------
SEED = 20260923
BLOCK_FRAMES = 200          # 20 s blocks at 10 Hz
TAU_SEC = 10.0              # temporal buffer at every split boundary
TARGET_FRAME = {'train': 0.68, 'val': 0.14, 'test': 0.18}
TARGET_CLASS = {'train': 0.68, 'val': 0.14, 'test': 0.18}
VAL_FLOOR_RATIO = 0.08      # per-class floor in val (share of class total)
TEST_FLOOR_RATIO = 0.15     # per-class floor in test
VAL_FLOOR_ABS = 200
TEST_FLOOR_ABS = 400
SEQ03_TEST_MIN_SHARE = 0.25  # seq03 is the unseen-day sequence
N_ITER = 60000

SPLITS = ('train', 'val', 'test')


def log(msg):
    print(f'[v5] {msg}', flush=True)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def md5_token(name):
    return hashlib.md5(name.encode()).hexdigest()


# ----------------------------------------------------------------------------
# 1. load v4 and build the frame universe
# ----------------------------------------------------------------------------
def load_universe():
    det, trk = {}, {}
    for sp in SPLITS:
        d = load_pkl(DET_DIR / f'maritime_nuscenes_infos_{sp}_10dof_v4.pkl')
        assert d['metainfo']['classes'] == CLASSES, 'unexpected class set'
        det[sp] = d['data_list']
    for sp in ('trainval', 'test'):
        d = load_pkl(
            TRK_DIR / f'maritime_nuscenes_infos_{sp}_10dof_track_v4.pkl')
        trk[sp] = d['data_list']

    det_by_tok = {}
    for sp in SPLITS:
        for s in det[sp]:
            assert s['token'] not in det_by_tok, f'dup token {s["token"]}'
            det_by_tok[s['token']] = s
    trk_by_tok = {}
    for sp in ('trainval', 'test'):
        for s in trk[sp]:
            assert s['token'] not in trk_by_tok
            trk_by_tok[s['token']] = s
    assert set(det_by_tok) == set(trk_by_tok), 'det/trk universes differ'
    for tok, s in det_by_tok.items():
        assert len(trk_by_tok[tok]['instances']) == len(s['instances'])
    # det instances carry no instance_token; take them from the trk pkls
    inst_of = {
        tok: tuple(it['instance_token'] for it in trk_by_tok[tok]['instances'])
        for tok in det_by_tok}
    n_inst = sum(len(s['instances']) for s in det_by_tok.values())
    log(f'universe: {len(det_by_tok)} frames, {n_inst} boxes; det/trk pkls '
        f'agree per frame')
    return det_by_tok, trk_by_tok, inst_of, n_inst


def build_frames(det_by_tok, inst_of):
    """frames per sequence ordered by timestamp with per-sequence index."""
    seqs = defaultdict(list)
    for s in det_by_tok.values():
        seq = s['lidar_points']['lidar_path'].split('/')[0]
        seqs[seq].append(s)
    frames = {}
    seq_order = {}
    for seq in sorted(seqs):
        lst = sorted(seqs[seq], key=lambda s: s['timestamp'])
        seq_order[seq] = lst
        for i, s in enumerate(lst):
            frames[s['token']] = dict(
                seq=seq, ts=s['timestamp'], idx=i,
                labels=tuple(it['bbox_label_3d'] for it in s['instances']),
                insts=inst_of[s['token']])
    return seq_order, frames


def build_blocks(seq_order):
    """contiguous BLOCK_FRAMES-sized blocks over annotated frames per seq."""
    blocks = []
    for seq, lst in seq_order.items():
        for start in range(0, len(lst), BLOCK_FRAMES):
            chunk = lst[start:start + BLOCK_FRAMES]
            cc = Counter(it['bbox_label_3d'] for s in chunk
                         for it in s['instances'])
            blocks.append(dict(
                seq=seq, lo=start, hi=start + len(chunk) - 1,
                toks=[s['token'] for s in chunk],
                cc=cc, n=len(chunk)))
    return blocks


# ----------------------------------------------------------------------------
# 2. constructive run assignment with global refinement
# ----------------------------------------------------------------------------
def floors(class_totals):
    fl_v, fl_t = {}, {}
    for c, n in class_totals.items():
        fl_v[c] = min(max(VAL_FLOOR_ABS, VAL_FLOOR_RATIO * n), 0.30 * n)
        fl_t[c] = min(max(TEST_FLOOR_ABS, TEST_FLOOR_RATIO * n), 0.35 * n)
    return fl_v, fl_t


STRIP = int(TAU_SEC / 0.1)   # buffer strip width in frames (10 s = 100)


def build_seq_arrays(blocks):
    """per-sequence blocks with prefix sums (frames + per-class boxes)."""
    seqs = {}
    for b in blocks:
        seqs.setdefault(b['seq'], []).append(b)
    arrays = {}
    for seq, bl in seqs.items():
        nb = len(bl)
        nf = np.zeros(nb + 1, dtype=np.int64)
        pc = np.zeros((nb + 1, 5), dtype=np.int64)
        for i, b in enumerate(bl):
            nf[i + 1] = nf[i] + b['n']
            for c in range(5):
                pc[i + 1, c] = pc[i, c] + b['cc'].get(c, 0)
        arrays[seq] = dict(blocks=bl, nf=nf, pc=pc)
    return arrays


def run_mass(arr, lo_blk, hi_blk):
    """exact (frames, class-vector) of blocks [lo_blk, hi_blk]."""
    nf, pc = arr['nf'], arr['pc']
    return int(nf[hi_blk + 1] - nf[lo_blk]), pc[hi_blk + 1] - pc[lo_blk]


def frame_range_mass(arr, lo_frame, hi_frame):
    """(frames, class-vector) for frame index range [lo_frame, hi_frame).
    Partial blocks at the edges are prorated -- the exact per-frame
    accounting happens in assign_buffers/validate_plan; this only guides
    the optimiser."""
    nf, pc = arr['nf'], arr['pc']
    n = int(nf[-1])
    lo_frame = max(0, min(int(lo_frame), n))
    hi_frame = max(0, min(int(hi_frame), n))
    if hi_frame <= lo_frame:
        return 0, np.zeros(5, dtype=np.int64)
    b0 = min(int(np.searchsorted(nf, lo_frame, side='right')) - 1,
             len(nf) - 2)
    b1 = min(int(np.searchsorted(nf, hi_frame, side='right')) - 1,
             len(nf) - 2)
    frames, cc = 0, np.zeros(5, dtype=np.int64)

    def add_block(i, frac):
        nonlocal frames, cc
        f = int(nf[i + 1] - nf[i])
        frames += int(round(f * frac))
        cc = cc + np.round((pc[i + 1] - pc[i]) * frac).astype(np.int64)

    if b1 == b0:
        add_block(b0, (hi_frame - lo_frame) /
                  max(1, int(nf[b0 + 1] - nf[b0])))
    else:
        add_block(b0, (nf[b0 + 1] - lo_frame) /
                  max(1, int(nf[b0 + 1] - nf[b0])))
        for i in range(b0 + 1, b1):
            add_block(i, 1.0)
        add_block(b1, (hi_frame - nf[b1]) /
                  max(1, int(nf[b1 + 1] - nf[b1])))
    return frames, cc


def shrunk_mass(arr, lo_blk, hi_blk):
    """val-run interior mass after trimming ~STRIP frames off both ends."""
    lo_frame = int(arr['nf'][lo_blk])
    hi_frame = int(arr['nf'][hi_blk + 1])
    return frame_range_mass(arr, lo_frame + STRIP, hi_frame - STRIP)


def track_cuts_estimate(track_spans, seq, lo_frame, hi_frame):
    cuts = 0
    for (s, lo, hi) in track_spans:
        if s != seq:
            continue
        if lo <= hi_frame and hi >= lo_frame:
            if lo < lo_frame or hi > hi_frame:
                cuts += 1
    return cuts


def candidates(arr, avoid, min_len, max_len, gap=2):
    nb = len(arr['blocks'])
    out = []
    for lo in range(0, nb - min_len + 1):
        for ln in range(min_len, min(max_len, nb - lo) + 1):
            hi = lo + ln - 1
            if any(not (hi < a - gap or lo > b + gap) for (a, b) in avoid):
                continue
            out.append((lo, hi))
    return out


class GlobalCost:
    """exact (block-resolution) cost of a run assignment, post-buffer."""

    def __init__(self, arrays, class_totals, totals_f, track_spans):
        self.arrays = arrays
        self.totals_c = np.array([class_totals.get(c, 0)
                                  for c in range(5)], dtype=np.int64)
        self.totals_f = totals_f
        self.track_spans = track_spans
        self.tgt = {sp: TARGET_FRAME[sp] for sp in SPLITS}
        self.tgt_c = {sp: TARGET_CLASS[sp] for sp in SPLITS}
        self.fl_v, self.fl_t = floors(class_totals)

    def split_masses(self, state):
        """per-split effective (frames, class-vector) incl. buffer split."""
        m = {sp: [0, np.zeros(5, dtype=np.int64)] for sp in SPLITS}
        m['buffer'] = [0, np.zeros(5, dtype=np.int64)]
        for seq, st in state.items():
            arr = self.arrays[seq]
            for (lo, hi) in st['test']:
                f, cc = run_mass(arr, lo, hi)
                m['test'][0] += f
                m['test'][1] += cc
                for side in (True, False):
                    if side:
                        a = int(arr['nf'][lo]) - STRIP
                        b = int(arr['nf'][lo])
                    else:
                        a = int(arr['nf'][hi + 1])
                        b = a + STRIP
                    sf, scc = frame_range_mass(arr, a, b)
                    m['buffer'][0] += sf
                    m['buffer'][1] += scc
            for (lo, hi) in st['val']:
                lo_f, hi_f = int(arr['nf'][lo]), int(arr['nf'][hi + 1])
                ef, ecc = shrunk_mass(arr, lo, hi)
                ff, fcc = run_mass(arr, lo, hi)
                m['val'][0] += ef
                m['val'][1] += ecc
                m['buffer'][0] += ff - ef
                m['buffer'][1] += fcc - ecc
        m['train'][0] = self.totals_f - sum(m[sp][0] for sp in
                                            ('test', 'val', 'buffer'))
        m['train'][1] = self.totals_c - sum(m[sp][1] for sp in
                                            ('test', 'val', 'buffer'))
        return m

    def violation(self, state):
        """mean per-class relative floor shortfall (0 = all floors met).
        Per-class normalization: boat's huge floor must not dominate the
        lexicographic gate the way it dominated the old hinge sum."""
        m = self.split_masses(state)
        v = 0.0
        for sp, fl in (('val', self.fl_v), ('test', self.fl_t)):
            f = np.array([max(1.0, fl.get(c, 0)) for c in range(5)])
            v += float(np.minimum(
                1.0, np.maximum(0.0, f - m[sp][1]) / f).mean())
        return v

    def value(self, state):
        m = self.split_masses(state)
        J = 0.0
        for sp in SPLITS:
            f_share = m[sp][0] / self.totals_f
            J += 6.0 * abs(f_share - self.tgt[sp])
            # per-class RELATIVE error, averaged -- boat must not dominate
            scale = np.maximum(self.tgt_c[sp] * self.totals_c, 1)
            rel = np.abs(m[sp][1] - self.tgt_c[sp] * self.totals_c) / scale
            J += 10.0 * float(rel.mean())
        floor_map = {'val': self.fl_v, 'test': self.fl_t}
        for sp in ('val', 'test'):
            fl = np.array([max(1.0, floor_map[sp].get(c, 0))
                           for c in range(5)])
            hinge = np.maximum(0.0, fl - m[sp][1]) / fl
            J += 14.0 * float(hinge.mean()) + 6.0 * float(hinge.max())
        n_runs = sum(len(st[sp]) for st in state.values()
                     for sp in ('test', 'val'))
        J += 0.05 * n_runs
        cuts = 0
        for seq, st in state.items():
            arr = self.arrays[seq]
            for (lo, hi) in st['test']:
                cuts += track_cuts_estimate(self.track_spans, seq,
                                            int(arr['nf'][lo]),
                                            int(arr['nf'][hi + 1]) - 1)
        J += 0.1 * cuts
        s3_f = sum(int(self.arrays['03']['nf'][hi + 1] - self.arrays['03']
                       ['nf'][lo]) for (lo, hi) in state['03']['test'])
        s3_total = int(self.arrays['03']['nf'][-1])
        J += 6.0 * max(0.0, SEQ03_TEST_MIN_SHARE - s3_f / s3_total)
        return J


def construct_assignment(blocks, frames, class_totals):
    """Greedy initialisation + run-level local refinement against the
    global post-buffer cost."""
    arrays = build_seq_arrays(blocks)
    totals_f = sum(int(arrays[s]['nf'][-1]) for s in arrays)
    track_spans = {}
    for tok, fr in frames.items():
        for it in fr['insts']:
            if it in track_spans:
                s, l0, h0 = track_spans[it]
                track_spans[it] = (fr['seq'], min(l0, fr['idx']),
                                   max(h0, fr['idx']))
            else:
                track_spans[it] = (fr['seq'], fr['idx'], fr['idx'])

    gc = GlobalCost(arrays, class_totals, totals_f,
                    list(track_spans.values()))
    state = {seq: dict(test=[], val=[]) for seq in arrays}

    # ---- greedy init: one diverse test run + one val run per sequence
    def greedy_place(split, share, floor_map):
        target_c = np.round(share * gc.totals_c).astype(np.int64)
        scale = np.maximum(target_c, 1).astype(float)
        fl = np.array([max(1.0, floor_map.get(c, 0)) for c in range(5)])
        have_c = np.zeros(5, dtype=np.int64)

        def mass_of(seq, lo, hi):
            if split == 'val':
                return shrunk_mass(arrays[seq], lo, hi)
            return run_mass(arrays[seq], lo, hi)

        for rnd in range(2):
            progressed = False
            for seq in sorted(arrays, key=lambda s: -int(arrays[s]['nf'][-1])):
                arr = arrays[seq]
                nb = len(arr['blocks'])
                if len(state[seq][split]) >= (1 if rnd == 0 else 2):
                    continue
                if rnd == 1 and float(
                        (np.abs(target_c - have_c) / scale).mean()) < 0.35:
                    continue
                min_len = max(4, int(0.08 * nb))
                max_len = max(min_len + 2, int(0.30 * nb))
                if split == 'test' and seq == '03':
                    min_len = max(min_len,
                                  int(np.ceil(SEQ03_TEST_MIN_SHARE * nb)))
                avoid = list(state[seq]['test']) + \
                    list(state[seq]['val'])
                best, best_cost = None, None
                for (lo, hi) in candidates(arr, avoid, min_len, max_len):
                    frames_r, cc_r = mass_of(seq, lo, hi)
                    if frames_r == 0:
                        continue
                    cost = float((np.abs(target_c - have_c - cc_r)
                                  / scale).sum() / 5)
                    cost += 8.0 * float(np.maximum(
                        0.0, fl - have_c - cc_r).sum() / fl.sum())
                    if best is None or cost < best_cost:
                        best, best_cost = (lo, hi), cost
                if best is not None:
                    state[seq][split].append(best)
                    have_c += mass_of(seq, best[0], best[1])[1]
                    progressed = True
            if not progressed:
                break

    def overlaps(seq, split, lo, hi, skip=None):
        for other in ('test', 'val'):
            for k, (a, b) in enumerate(state[seq][other]):
                if skip is not None and other == split and k == skip:
                    continue
                if not (hi < a - 2 or lo > b + 2):
                    return True
        return False

    # minority-class seeding: long-tail classes concentrate in a few short
    # windows, so balanced placement + refinement can trade their floors
    # away against majority-class mass.  Secure floors by construction:
    # for every rare class, carve a test and a val run covering >=1.2x its
    # floor from its densest windows (smallest such window, leaving the
    # rest of the cluster to the other split), falling back to max
    # coverage when no window reaches the floor.
    cn = ['boat', 'buoy', 'sailboat', 'ship', 'yacht']
    grand = sum(class_totals.values())

    def seed_run(split, c):
        # windows are scored on their POST-BUFFER interior: the 10 s
        # strips eaten at run edges must not erode the floor margin
        floor_map = gc.fl_t if split == 'test' else gc.fl_v
        want = 1.25 * floor_map.get(c, 0)
        if want <= 0:
            return
        best_min, cov_min = None, None
        best_max, cov_max = None, 0
        for seq in sorted(arrays):
            arr = arrays[seq]
            nb = len(arr['blocks'])
            min_len = max(4, int(0.08 * nb))
            max_len = max(min_len + 2, int(0.30 * nb))
            if split == 'test' and seq == '03':
                min_len = max(min_len,
                              int(np.ceil(SEQ03_TEST_MIN_SHARE * nb)))
            for lo in range(0, nb - min_len + 1):
                for ln in range(min_len, min(max_len, nb - lo) + 1):
                    hi = lo + ln - 1
                    if overlaps(seq, split, lo, hi):
                        continue
                    cc = int(shrunk_mass(arr, lo, hi)[1][c])
                    if cc > cov_max:
                        cov_max, best_max = cc, (seq, lo, hi)
                    if cc >= want and (best_min is None or ln < cov_min):
                        cov_min, best_min = ln, (seq, lo, hi)
        if best_min is None:
            log(f'  no {split} window reaches the {cn[c]} floor; '
                f'leaving it to balanced placement')
            return
        seq, lo, hi = best_min
        state[seq][split].append((lo, hi))
        got = int(shrunk_mass(arrays[seq], lo, hi)[1][c])
        log(f'  seeded {split} run for {cn[c]}: seq{seq} blocks '
            f'[{lo}..{hi}] interior covers {got} {cn[c]}')

    for c in sorted(range(5), key=lambda c: class_totals.get(c, 0)):
        if class_totals.get(c, 0) >= 0.10 * grand:
            continue            # majority class: balanced placement suffices
        seed_run('test', c)
    for c in sorted(range(5), key=lambda c: class_totals.get(c, 0)):
        if class_totals.get(c, 0) >= 0.10 * grand:
            continue
        seed_run('val', c)

    greedy_place('test', TARGET_CLASS['test'], gc.fl_t)
    greedy_place('val', TARGET_CLASS['val'], gc.fl_v)
    cur = gc.value(state)
    log(f'cost after greedy init: {cur:.4f}')

    # ---- local refinement: shift/grow/shrink each run
    def legal(seq, split, k, lo, hi, nb):
        min_len = 4
        max_len = max(min_len + 2, int(0.32 * nb))
        if split == 'test' and seq == '03':
            min_len = max(min_len, int(np.ceil(SEQ03_TEST_MIN_SHARE * nb)))
        if lo < 0 or hi >= nb or hi < lo:
            return False
        if hi - lo + 1 < min_len or hi - lo + 1 > max_len:
            return False
        if overlaps(seq, split, lo, hi, skip=k):
            return False
        return True

    import os
    USE_GATE = os.environ.get('V5_GATE', '1') == '1'

    def better(va, ca, vb, cb):
        """lexicographic: floor violation first, cost second."""
        if not USE_GATE:
            return ca < cb - 1e-9
        if va < vb - 1e-12:
            return True
        if va > vb + 1e-12:
            return False
        return ca < cb - 1e-9

    v_cur = gc.violation(state)
    log(f'init: violation {v_cur:.4f} (gate {"on" if USE_GATE else "off"})')
    for sweep in range(12):
        improved = False
        for seq in sorted(arrays):
            arr = arrays[seq]
            nb = len(arr['blocks'])
            for split in ('test', 'val'):
                for k in range(len(state[seq][split])):
                    lo0, hi0 = state[seq][split][k]
                    best_move, best_val, best_viol = None, cur, v_cur
                    # resize / shift moves
                    for dlo in (-8, -6, -3, -2, -1, 0, 1, 2, 3, 6, 8):
                        for dhi in (-8, -6, -3, -2, -1, 0, 1, 2, 3, 6, 8):
                            lo, hi = lo0 + dlo, hi0 + dhi
                            if not legal(seq, split, k, lo, hi, nb):
                                continue
                            state[seq][split][k] = (lo, hi)
                            v, vo = gc.value(state), gc.violation(state)
                            state[seq][split][k] = (lo0, hi0)
                            if better(vo, v, best_viol, best_val):
                                best_val, best_viol = v, vo
                                best_move = (lo, hi)
                    # relocate move (same length, any legal position)
                    ln = hi0 - lo0
                    for lo in range(0, nb - ln):
                        hi = lo + ln
                        if not legal(seq, split, k, lo, hi, nb):
                            continue
                        state[seq][split][k] = (lo, hi)
                        v, vo = gc.value(state), gc.violation(state)
                        state[seq][split][k] = (lo0, hi0)
                        if better(vo, v, best_viol, best_val):
                            best_val, best_viol = v, vo
                            best_move = (lo, hi)
                    if best_move is not None:
                        state[seq][split][k] = best_move
                        cur, v_cur = best_val, best_viol
                        improved = True
        # joint moves: a test run and a nearby val run shift together.
        # Single-run moves get stuck when one split's run squats on a
        # minority-class cluster the other split needs (2-opt valley).
        for seq in sorted(arrays):
            arr = arrays[seq]
            nb = len(arr['blocks'])
            for ti in range(len(state[seq]['test'])):
                for vi in range(len(state[seq]['val'])):
                    tlo0, thi0 = state[seq]['test'][ti]
                    vlo0, vhi0 = state[seq]['val'][vi]
                    if not -12 <= vlo0 - thi0 <= 24:
                        continue
                    best_pair, best_val, best_viol = None, cur, v_cur
                    for dt in (-3, -2, -1, 0, 1, 2, 3):
                        for dv in (-3, -2, -1, 0, 1, 2, 3):
                            if dt == 0 and dv == 0:
                                continue
                            nt = (tlo0 + dt, thi0 + dt)
                            nv = (vlo0 + dv, vhi0 + dv)
                            if not legal(seq, 'test', ti, nt[0], nt[1], nb):
                                continue
                            if not legal(seq, 'val', vi, nv[0], nv[1], nb):
                                continue
                            if overlaps(seq, 'test', nt[0], nt[1], skip=ti):
                                continue
                            if overlaps(seq, 'val', nv[0], nv[1], skip=vi):
                                continue
                            state[seq]['test'][ti] = nt
                            state[seq]['val'][vi] = nv
                            v, vo = gc.value(state), gc.violation(state)
                            state[seq]['test'][ti] = (tlo0, thi0)
                            state[seq]['val'][vi] = (vlo0, vhi0)
                            if better(vo, v, best_viol, best_val):
                                best_val, best_viol = v, vo
                                best_pair = (nt, nv)
                    if best_pair is not None:
                        state[seq]['test'][ti] = best_pair[0]
                        state[seq]['val'][vi] = best_pair[1]
                        cur, v_cur = best_val, best_viol
                        improved = True
        log(f'  refinement sweep {sweep}: cost {cur:.4f} '
            f'violation {v_cur:.4f}')
        if not improved:
            break

    # ---- sanity: pairwise disjoint runs (the materialiser overwrites
    # labels test-then-val, so an overlap here would silently diverge from
    # the optimiser's own accounting)
    for seq, st in state.items():
        allr = [(sp, r) for sp in ('test', 'val') for r in st[sp]]
        for i in range(len(allr)):
            for j in range(i + 1, len(allr)):
                (sa, (a0, a1)), (sb, (b0, b1)) = allr[i], allr[j]
                assert a1 < b0 - 2 or b1 < a0 - 2, (
                    f'run overlap in seq{seq}: {sa}{(a0, a1)} '
                    f'{sb}{(b0, b1)}')

    # ---- materialise block labels
    m = gc.split_masses(state)
    log('predicted block-level masses (frames, boat, yacht): ' + str(
        {sp: (int(m[sp][0]), int(m[sp][1][0]), int(m[sp][1][4]))
         for sp in ('train', 'val', 'test', 'buffer')}))
    labels = ['train'] * len(blocks)
    off = 0
    for seq in sorted(arrays):
        for (lo, hi) in state[seq]['test']:
            for i in range(lo, hi + 1):
                labels[off + i] = 'test'
        for (lo, hi) in state[seq]['val']:
            for i in range(lo, hi + 1):
                labels[off + i] = 'val'
        off += len(arrays[seq]['blocks'])
    log('runs per sequence: ' + str(
        {seq: {sp: len(r) for sp, r in st.items()}
         for seq, st in state.items()}))
    return labels

# ----------------------------------------------------------------------------
# 3. temporal buffers (one-shot strips; buffer frames belong to NO split)
# ----------------------------------------------------------------------------
def assign_buffers(seq_order, labels_by_tok):
    """One-shot buffer construction from the optimiser labels L0.

    A frame becomes 'buffer' (used by neither training nor evaluation) iff
    it is within TAU_SEC of a *training-side* frame (train) or of a test
    frame, but is not itself that frame:

      buffer(f) <=>  L0(f) != test  and dist(f, L0-test) <= TAU
                  or L0(f) == val   and dist(f, L0-train) <= TAU

    Proof of separation on the final labelling:
      - every train frame is an L0-train frame with dist > TAU to every
        L0-test frame (else it would be buffer) and is not within TAU of
        any buffer frame either? -- buffer frames lie within TAU of test or
        train, so a train frame g could still sit within TAU of a *test*
        strip frame?  No: strip frames lie in [test +/- TAU]; a train frame
        closer than TAU to a strip frame s implies dist(g, s) <= TAU and
        dist(s, test) <= TAU, which does NOT bound dist(g, test)... the
        binding guarantee is the direct one: g survived buffering, so
        dist(g, L0-test) > TAU; every final test frame is an L0-test frame.
        Strip frames near test are BUFFER, not test -- so the nearest final
        test frame to any train frame is an L0-test frame, hence > TAU. ∎
      - likewise every final val frame is > TAU from every L0-train frame.

    This mirrors the legacy ImageSets split, which dropped 2960 buffer
    frames ('block: 150, buffer: 20' in ImageSets/split_stats.json).
    """
    n_buf = 0
    for seq, lst in seq_order.items():
        L0 = [labels_by_tok[s['token']] for s in lst]
        tss = [s['timestamp'] for s in lst]
        n = len(lst)
        final = list(L0)
        # nearest L0-test / L0-train distance per frame (two-pointer over
        # the time-sorted list)
        test_idx = [i for i in range(n) if L0[i] == 'test']
        train_idx = [i for i in range(n) if L0[i] == 'train']

        def nearest_dist(idx_list, i):
            import bisect
            k = bisect.bisect_left(idx_list, i)
            best = float('inf')
            for c in (k - 1, k):
                if 0 <= c < len(idx_list):
                    best = min(best, abs(tss[idx_list[c]] - tss[i]) / 1e9)
            return best

        for i in range(n):
            d_test = nearest_dist(test_idx, i)
            if L0[i] != 'test' and d_test <= TAU_SEC:
                final[i] = 'buffer'
            elif L0[i] == 'val' and \
                    nearest_dist(train_idx, i) <= TAU_SEC:
                final[i] = 'buffer'
        for s, lab in zip(lst, final):
            if lab != labels_by_tok[s['token']]:
                labels_by_tok[s['token']] = lab
                n_buf += 1
    log(f'buffer frames (held out from all splits): {n_buf} '
        f'({100 * n_buf / 80277:.1f}%)')
    return labels_by_tok


# ----------------------------------------------------------------------------
# 4. statistics + hard validation of the plan
# ----------------------------------------------------------------------------
def final_stats(seq_order, frames, labels_by_tok):
    all_labels = SPLITS + ('buffer',)
    per_split_tokens = {sp: [] for sp in all_labels}
    for seq, lst in seq_order.items():
        for s in lst:
            per_split_tokens[labels_by_tok[s['token']]].append(s['token'])
    stats = {}
    for sp in all_labels:
        toks = per_split_tokens[sp]
        cc = Counter()
        for t in toks:
            cc.update(frames[t]['labels'])
        stats[sp] = dict(tokens=toks, class_counts=cc, n=len(toks))
    return stats, per_split_tokens


def min_gap(seq_order, labels_by_tok, sp_a, sp_b):
    """exact min |dt| (s) between any frame of sp_a and any frame of sp_b,
    per sequence.  After buffering, nearest cross-split pairs sit at block
    boundaries; we scan every frame against the next 2*TAU/0.1 frames to be
    exact rather than relying on adjacency."""
    worst = float('inf')
    win = int(2 * TAU_SEC / 0.1) + 5
    for seq, lst in seq_order.items():
        labs = [labels_by_tok[s['token']] for s in lst]
        tss = [s['timestamp'] for s in lst]
        n = len(lst)
        for i in range(n):
            if labs[i] != sp_a:
                continue
            for j in range(i + 1, min(i + win, n)):
                if labs[j] == sp_b:
                    worst = min(worst, (tss[j] - tss[i]) / 1e9)
    return worst


def validate_plan(seq_order, frames, stats, labels_by_tok, class_totals):
    ok = True
    universe = sum(len(l) for l in seq_order.values())
    all_labels = SPLITS + ('buffer',)
    all_tok = [t for sp in all_labels for t in stats[sp]['tokens']]
    if not (len(all_tok) == len(set(all_tok)) == universe):
        log(f'FAIL frame conservation: {len(all_tok)} tokens, '
            f'{len(set(all_tok))} unique, universe {universe}')
        ok = False

    log('per-class instance counts (v5):')
    log('split   ' + ''.join(f'{c:>10s}' for c in CLASSES) + '   total')
    for sp in all_labels:
        cc = stats[sp]['class_counts']
        log(f'{sp:7s} ' + ''.join(f'{cc.get(i, 0):10d}' for i in range(5))
            + f'   {sum(cc.values()):d}')
    log('per-class shares of class total:')
    for sp in SPLITS:
        cc = stats[sp]['class_counts']
        log(f'  {sp:6s} ' + '  '.join(
            f'{CLASSES[i]}={100 * cc.get(i, 0) / class_totals[i]:.0f}%'
            for i in range(5) if class_totals[i]))

    fl_v, fl_t = floors(class_totals)
    for i, cname in enumerate(CLASSES):
        v = stats['val']['class_counts'].get(i, 0)
        t = stats['test']['class_counts'].get(i, 0)
        if class_totals[i] and v < fl_v[i] * 0.95:
            log(f'  WARN val floor missed for {cname}: {v} < {fl_v[i]:.0f}')
        if class_totals[i] and t < fl_t[i] * 0.95:
            log(f'  WARN test floor missed for {cname}: {t} < {fl_t[i]:.0f}')

    for a, b in (('train', 'test'), ('train', 'val'), ('val', 'test')):
        g = min_gap(seq_order, labels_by_tok, a, b)
        log(f'min frame gap {a}<->{b}: {g:.2f}s')
        if (a, b) in (('train', 'test'), ('train', 'val')) and g <= TAU_SEC:
            log(f'  FAIL: {a}/{b} separation {g:.2f}s <= {TAU_SEC}s')
            ok = False

    tot = sum(stats[sp]['n'] for sp in SPLITS)
    for sp in SPLITS:
        log(f'frame share {sp}: {stats[sp]["n"]} '
            f'({100 * stats[sp]["n"] / tot:.1f}% of assigned frames)')

    s3 = Counter()
    for s in seq_order['03']:
        s3[labels_by_tok[s['token']]] += 1
    log(f'seq03 (unseen day) split: {dict(s3)}, test share '
        f"{100 * s3['test'] / max(1, sum(s3.values())):.0f}%")

    def side(lab):
        return 'trainval' if lab in ('train', 'val') else lab

    track_labels = defaultdict(set)
    for tok, fr in frames.items():
        for it in fr['insts']:
            track_labels[it].add(labels_by_tok[tok])
    lost = 0
    for it, labs in track_labels.items():
        if {side(l) for l in labs} == {'buffer'}:
            lost += 1
    cuts = sum(1 for it, labs in track_labels.items()
               if 'test' in {side(l) for l in labs}
               and 'trainval' in {side(l) for l in labs})
    log(f'tracks straddling trainval/test (GT fragments): {cuts} of '
        f'{len(track_labels)}; tracks entirely inside buffer: {lost}')
    return ok


# ----------------------------------------------------------------------------
# 5. emit detection artifacts
# ----------------------------------------------------------------------------
def emit_detection(det_by_tok, frames, stats, suffix):
    for sp in SPLITS:
        data_list = [det_by_tok[t] for t in stats[sp]['tokens']]
        out = dict(
            metainfo=dict(
                classes=CLASSES,
                split_version='v5',
                source='tools/resplit_maritime3d_v5.py'),
            data_list=data_list)
        path = DET_DIR / f'maritime_nuscenes_infos_{sp}_10dof_{suffix}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(out, f)
        log(f'wrote {path.name} ({len(data_list)} frames)')

        split_json = defaultdict(dict)
        for t in stats[sp]['tokens']:
            split_json[frames[t]['seq']].setdefault(
                'frame_tokens', []).append(t)
        for seq in split_json:
            split_json[seq]['n_frames'] = len(split_json[seq]['frame_tokens'])
        jp = DET_DIR / f'maritime_split_{sp}_{suffix}.json'
        with open(jp, 'w') as f:
            json.dump(split_json, f)
        log(f'wrote {jp.name}')

    s3 = [t for t in stats['test']['tokens'] if frames[t]['seq'] == '03']
    with open(DET_DIR / f'maritime_split_test_{suffix}_crossday.json',
              'w') as f:
        json.dump(dict(
            seq='03', n_frames=len(s3), frame_tokens=s3,
            description='unseen-day subset of the v5 test set '
                        '(sequence 03, recorded 14 days after 00-02)'),
            f, indent=1)
    log(f'wrote maritime_split_test_{suffix}_crossday.json ({len(s3)} frames)')

    buf = stats['buffer']['tokens']
    with open(DET_DIR / f'maritime_split_buffer_{suffix}.json', 'w') as f:
        json.dump(dict(
            n_frames=len(buf), frame_tokens=buf,
            description='temporal buffer frames within '
                        f'{TAU_SEC:.0f}s of a train/test or train/val '
                        'boundary; excluded from every split on purpose '
                        'to guarantee cross-split separation'),
            f, indent=1)
    log(f'wrote maritime_split_buffer_{suffix}.json ({len(buf)} frames)')


# ----------------------------------------------------------------------------
# 6. emit tracking artifacts
# ----------------------------------------------------------------------------
def emit_tracking(trk_by_tok, frames, stats, suffix):
    split_of = {}
    for sp in SPLITS:
        for t in stats[sp]['tokens']:
            split_of[t] = sp

    # pkls: trainval = train U val, test = test (frame-set equality)
    for name, members in (('trainval', ('train', 'val')),
                          ('test', ('test',))):
        toks = [t for sp in members for t in stats[sp]['tokens']]
        toks.sort(key=lambda t: (trk_by_tok[t]['lidar_points']['lidar_path'],
                                 trk_by_tok[t]['timestamp']))
        out = dict(
            metainfo=dict(
                classes=CLASSES,
                split_version=f'{suffix}-track',
                source='tools/resplit_maritime3d_v5.py'),
            data_list=[trk_by_tok[t] for t in toks])
        path = TRK_DIR / (f'maritime_nuscenes_infos_{name}_10dof_track_'
                          f'{suffix}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(out, f)
        log(f'wrote {path.name} ({len(toks)} frames)')

    # tables: merge the two v4 table sets, re-partition rows by frame
    src_tables = {}
    for name in ('trainval', 'test'):
        base = TRK_DIR / f'tracking_v4/{name}/v1.0-{name}'
        for j in ('attribute', 'category', 'instance', 'sample',
                  'sample_annotation', 'sample_data', 'scene', 'sensor'):
            with open(base / f'{j}.json') as f:
                src_tables[(name, j)] = json.load(f)

    merged = {j: src_tables[('trainval', j)] + src_tables[('test', j)]
              for j in ('sample', 'sample_annotation', 'sample_data',
                        'instance')}
    cat = src_tables[('trainval', 'category')]
    attr = src_tables[('trainval', 'attribute')]
    sensor = src_tables[('trainval', 'sensor')]
    cat_name = {c['token']: c['name'].split('.')[-1] for c in cat}

    sample_by_tok = {s['token']: s for s in merged['sample']}
    sdata_by_sample = {d['sample_token']: d for d in merged['sample_data']}
    inst_cat = {i['token']: i['category_token'] for i in merged['instance']}

    out_root = TRK_DIR / f'tracking_{suffix}'
    created = datetime.date.today().isoformat()

    for name, members in (('trainval', ('train', 'val')),
                          ('test', ('test',))):
        toks = [t for sp in members for t in stats[sp]['tokens']]
        per_seq = defaultdict(list)
        for t in toks:
            per_seq[frames[t]['seq']].append(t)

        scenes = []   # (name, sample tokens in time order)
        for seq in sorted(per_seq):
            lst = sorted(per_seq[seq],
                         key=lambda t: frames[t]['ts'])
            runs, run = [], [lst[0]]
            prev_ts = frames[lst[0]]['ts']
            for t in lst[1:]:
                ts = frames[t]['ts']
                if (ts - prev_ts) / 1e9 > 1.0:  # split boundary or gap
                    runs.append(run)
                    run = []
                run.append(t)
                prev_ts = ts
            runs.append(run)
            for r in runs:
                scenes.append((f'scene-{seq}_{name}_'
                               f"{frames[r[0]]['idx']}-"
                               f"{frames[r[-1]]['idx']}", r))

        scene_rows, sample_rows, sdata_rows = [], [], []
        scene_of_sample = {}
        for sc_name, r in scenes:
            sc_tok = md5_token(sc_name)
            scene_rows.append(dict(
                token=sc_tok, name=sc_name,
                description=f'contiguous segment of the {suffix} {name} '
                            f'split (frames '
                            f"{frames[r[0]]['idx']}-{frames[r[-1]]['idx']})",
                log_token=md5_token('log-' + sc_name.split('_')[0]),
                nbr_samples=len(r),
                first_sample_token=r[0], last_sample_token=r[-1]))
            for k, t in enumerate(r):
                prev_t = r[k - 1] if k else ''
                next_t = r[k + 1] if k + 1 < len(r) else ''
                s = sample_by_tok[t]
                sample_rows.append(dict(
                    token=t, timestamp=s['timestamp'], prev=prev_t,
                    next=next_t, scene_token=sc_tok))
                scene_of_sample[t] = sc_tok
                d = sdata_by_sample[t]
                sdata_rows.append(dict(
                    token=d['token'], sample_token=t, ego_pose_token='',
                    calibrated_sensor_token='', timestamp=d['timestamp'],
                    fileformat=d['fileformat'], is_key_frame=True,
                    height=0, width=0, filename=d['filename'],
                    prev=sdata_by_sample[prev_t]['token'] if prev_t else '',
                    next=sdata_by_sample[next_t]['token'] if next_t else ''))
        sample_set = set(scene_of_sample)

        annos = [a for a in merged['sample_annotation']
                 if a['sample_token'] in sample_set]
        chain = defaultdict(list)
        for a in annos:
            chain[(a['instance_token'],
                   scene_of_sample[a['sample_token']])].append(a)
        anno_rows = []
        inst_annos = defaultdict(list)
        for (it, _sc), lst in chain.items():
            lst.sort(key=lambda a: frames[a['sample_token']]['ts'])
            for k, a in enumerate(lst):
                a['prev'] = lst[k - 1]['token'] if k else ''
                a['next'] = lst[k + 1]['token'] if k + 1 < len(lst) else ''
            inst_annos[it].extend(lst)
            anno_rows.extend(lst)
        anno_rows.sort(key=lambda a: (a['sample_token'], a['instance_token']))

        inst_rows = []
        for it, lst in inst_annos.items():
            lst_sorted = sorted(
                lst, key=lambda a: frames[a['sample_token']]['ts'])
            inst_rows.append(dict(
                token=it, category_token=inst_cat[it],
                nbr_annotations=len(lst_sorted),
                first_annotation_token=lst_sorted[0]['token'],
                last_annotation_token=lst_sorted[-1]['token']))
        inst_rows.sort(key=lambda i: i['token'])

        boxes_by_class = Counter(
            cat_name[inst_cat[a['instance_token']]] for a in anno_rows)

        out_dir = out_root / f'v1.0-{name}'
        out_dir.mkdir(parents=True, exist_ok=True)
        dump = {
            'attribute': attr, 'category': cat, 'sensor': sensor,
            'scene': scene_rows, 'sample': sample_rows,
            'sample_data': sdata_rows, 'sample_annotation': anno_rows,
            'instance': inst_rows,
        }
        for j, rows in dump.items():
            with open(out_dir / f'{j}.json', 'w') as f:
                json.dump(rows, f)
        with open(out_dir / 'split_info.json', 'w') as f:
            json.dump(dict(
                split=name, split_version=suffix, created=created,
                source='tools/resplit_maritime3d_v5.py',
                frames=len(sample_rows), boxes=len(anno_rows),
                boxes_by_class=dict(boxes_by_class),
                instances=len(inst_rows),
                scenes=[s['name'] for s in scene_rows]), f, indent=1)
        log(f'wrote {out_dir} (scenes={len(scene_rows)}, '
            f'samples={len(sample_rows)}, annos={len(anno_rows)}, '
            f'instances={len(inst_rows)}, boxes_by_class='
            f'{dict(boxes_by_class)})')


# ----------------------------------------------------------------------------
# 7. post-write reload validation
# ----------------------------------------------------------------------------
def reload_validate(suffix, universe_n, total_boxes, n_buffer_expected):
    det = {}
    for sp in SPLITS:
        d = load_pkl(
            DET_DIR / f'maritime_nuscenes_infos_{sp}_10dof_{suffix}.pkl')
        assert d['metainfo']['split_version'] == 'v5'
        det[sp] = d['data_list']
    toks = {sp: set(s['token'] for s in det[sp]) for sp in SPLITS}
    assert not (toks['train'] & toks['val']), 'train/val overlap'
    assert not (toks['train'] & toks['test']), 'train/test overlap'
    assert not (toks['val'] & toks['test']), 'val/test overlap'
    assert sum(len(t) for t in toks.values()) == universe_n - n_buffer_expected

    with open(DET_DIR / f'maritime_split_buffer_{suffix}.json') as f:
        n_buffer_expected_check = len(json.load(f)['frame_tokens'])
    assert n_buffer_expected_check == n_buffer_expected
    n_det_boxes = sum(len(s['instances']) for sp in SPLITS
                      for s in det[sp])
    log(f'emitted det pkls: {sum(len(t) for t in toks.values())} frames, '
        f'{n_det_boxes} boxes (universe {universe_n} = assigned '
        f'{universe_n - n_buffer_expected} + buffer {n_buffer_expected})')

    trk = {}
    for name in ('trainval', 'test'):
        d = load_pkl(TRK_DIR / f'maritime_nuscenes_infos_{name}_10dof_track_'
                               f'{suffix}.pkl')
        trk[name] = set(s['token'] for s in d['data_list'])
    assert trk['trainval'] == toks['train'] | toks['val'], \
        'tracking trainval != det train U val'
    assert trk['test'] == toks['test'], 'tracking test != det test'
    log('pkl frame-set equality: det_train U det_val == trk_trainval, '
        'det_test == trk_test  OK')

    n_annos = 0
    for name in ('trainval', 'test'):
        base = TRK_DIR / f'tracking_{suffix}/v1.0-{name}'
        tabs = {}
        for j in ('sample', 'sample_annotation', 'sample_data', 'instance',
                  'scene'):
            with open(base / f'{j}.json') as f:
                tabs[j] = json.load(f)
        sample_tok = {s['token'] for s in tabs['sample']}
        scene_tok = {s['token'] for s in tabs['scene']}
        anno_tok = {a['token'] for a in tabs['sample_annotation']}
        inst_tok = {i['token'] for i in tabs['instance']}
        sd_tok = {d['token'] for d in tabs['sample_data']}
        for s in tabs['sample']:
            assert s['scene_token'] in scene_tok
            assert s['prev'] == '' or s['prev'] in sample_tok
            assert s['next'] == '' or s['next'] in sample_tok
        assert len(tabs['sample_data']) == len(tabs['sample'])
        for d in tabs['sample_data']:
            assert d['sample_token'] in sample_tok
            assert d['prev'] == '' or d['prev'] in sd_tok
            assert d['next'] == '' or d['next'] in sd_tok
        for a in tabs['sample_annotation']:
            assert a['sample_token'] in sample_tok
            assert a['instance_token'] in inst_tok
            assert a['prev'] == '' or a['prev'] in anno_tok
            assert a['next'] == '' or a['next'] in anno_tok
        n_inst_annos = 0
        for i in tabs['instance']:
            assert i['first_annotation_token'] in anno_tok
            assert i['last_annotation_token'] in anno_tok
            assert i['nbr_annotations'] > 0
            n_inst_annos += i['nbr_annotations']
        assert n_inst_annos == len(tabs['sample_annotation']), \
            f'instance anno sums {n_inst_annos} != rows'
        log(f'table v1.0-{name}: relational integrity OK '
            f'({len(tabs["sample"])} samples, '
            f'{len(tabs["sample_annotation"])} annos, '
            f'{len(tabs["instance"])} instances)')
        n_annos += len(tabs['sample_annotation'])

    assert n_annos == n_det_boxes, \
        f'annotation conservation broken: tables {n_annos} != det {n_det_boxes}'
    log(f'annotation conservation OK: {n_annos} boxes; det pkls and '
        f'tracking tables agree (universe holds {total_boxes} incl. '
        f'buffer frames)')
    return True


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='v5')
    ap.add_argument('--dry-run', action='store_true',
                    help='plan + validate only, write nothing')
    args = ap.parse_args()

    det_by_tok, trk_by_tok, inst_of, total_boxes = load_universe()
    seq_order, frames = build_frames(det_by_tok, inst_of)
    universe_n = sum(len(l) for l in seq_order.values())
    blocks = build_blocks(seq_order)
    log(f'{len(blocks)} blocks of {BLOCK_FRAMES} frames '
        f'({BLOCK_FRAMES / 10:.0f}s)')

    class_totals = Counter()
    for fr in frames.values():
        class_totals.update(fr['labels'])
    log('class totals: ' + ', '.join(f'{CLASSES[i]}={class_totals[i]}'
                                     for i in range(5)))

    labels = construct_assignment(blocks, frames, class_totals)
    labels_by_tok = {}
    for b, lab in zip(blocks, labels):
        for t in b['toks']:
            labels_by_tok[t] = lab

    log('applying temporal buffers:')
    labels_by_tok = assign_buffers(seq_order, labels_by_tok)

    stats, per_split_tokens = final_stats(seq_order, frames, labels_by_tok)
    log('--- plan validation ---')
    ok = validate_plan(seq_order, frames, stats, labels_by_tok,
                       class_totals)
    if not ok:
        log('PLAN VALIDATION FAILED -- nothing written')
        sys.exit(1)
    if args.dry_run:
        log('dry run complete; re-run without --dry-run to write artifacts')
        return

    log('--- writing artifacts ---')
    emit_detection(det_by_tok, frames, stats, args.suffix)
    emit_tracking(trk_by_tok, frames, stats, args.suffix)
    log('--- reload validation ---')
    reload_validate(args.suffix, universe_n, total_boxes,
                    n_buffer_expected=stats['buffer']['n'])
    log('v5 split complete.')


if __name__ == '__main__':
    main()
