#!/usr/bin/env python3
"""Mirror every Maritime3D run into TensorBoard event files.

    python3 tools/tb_sync.py [--loop 120] [--runs bench_gn_*]

For each work_dirs/<run>: the training scalars of every vis_data/scalars.json
(loss, lr, grad_norm, ... vs iteration) and the TRUE per-evaluation val
metrics recovered from the log tables (tools/maritime_val_history.py; the
logged val summaries of runs before 2026-09-29 are running means) go to
work_dirs/tb/<run>/. Event files are rewritten from scratch each pass, so
the result is idempotent. Then: tensorboard --logdir work_dirs/tb
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maritime_val_history import find_log, parse_log  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WD = os.path.join(ROOT, 'work_dirs')
SKIP = {'lr', 'base_lr', 'epoch', 'iter', 'step', 'memory', 'time',
        'data_time'}


def sync_run(run):
    files = sorted(glob.glob(os.path.join(WD, run, '*', 'vis_data',
                                          '*scalars.json')))
    if not files:
        return False
    out = os.path.join(WD, 'tb', run)
    shutil.rmtree(out, ignore_errors=True)
    w = SummaryWriter(out)
    n = 0
    for f in files:
        for line in open(f, errors='replace'):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if 'loss' not in r:
                continue  # val summaries: taken from the tables below
            it = r.get('step', r.get('iter'))
            for k, v in r.items():
                if k in SKIP or not isinstance(v, (int, float)):
                    continue
                w.add_scalar(f'train/{k}', v, it)
            for k in ('lr', 'grad_norm', 'time', 'memory'):
                if k in r:
                    w.add_scalar(f'sys/{k}', r[k], it)
            n += 1
    log = find_log(os.path.join(WD, run))
    curve = parse_log(log) if log else {}
    for ep in sorted(k for k in curve if isinstance(k, int)):
        for k, v in curve[ep].items():
            if v is not None:
                w.add_scalar(f'val/{k}', v, ep)
    w.close()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='*', default=['bench_gn_*'])
    ap.add_argument('--loop', type=int, default=0,
                    help='seconds between passes (0 = once)')
    args = ap.parse_args()
    while True:
        runs = sorted({os.path.basename(p) for pat in args.runs
                       for p in glob.glob(os.path.join(WD, pat))
                       if os.path.isdir(p)})
        for run in runs:
            try:
                sync_run(run)
            except Exception as e:  # keep the loop alive
                print(run, 'failed:', e, file=sys.stderr)
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == '__main__':
    main()
