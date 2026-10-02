#!/usr/bin/env python3
"""True per-epoch val metrics of mmengine runs on the maritime benchmark.

The summary line mmengine logs after each evaluation is NOT the value of that
evaluation for runs trained before 2026-09-29: LogProcessor averages every
scalar matching ``.*(loss|time|data_time|grad_norm).*`` over a 50-entry
window, and "Mari*time*/..." matches, so each logged metric was the mean of
all evaluations so far (the best-checkpoint hook used the true values). The
tables that MaritimeMetric prints itself are per evaluation, so this parses
those.

    python3 tools/maritime_val_history.py work_dirs/bench_gn_cp [...]
    python3 tools/maritime_val_history.py --json out.json work_dirs/...
"""
import argparse
import glob
import json
import os
import re

HDR = re.compile(r'Maritime3D detection results \(AP40, IoU')
DIST_HDR = re.compile(r'Maritime3D detection results \(AP40, BEV centre')
EPOCH = re.compile(r'Epoch\((val|test)\) +\[(\d+)\]')
ROW = re.compile(r'^\|\s*([\w.-]+)\s*\|(.*)\|\s*$')


def _table(lines, i):
    """Rows of the +---+ table starting after line i: header, rows."""
    rows = []
    j = i + 1
    while j < len(lines) and not lines[j].startswith('+'):
        j += 1
    while j < len(lines) and (lines[j].startswith('+')
                              or lines[j].startswith('|')):
        m = ROW.match(lines[j].rstrip())
        if m:
            rows.append([m.group(1)] + [c.strip() for c in
                                         m.group(2).split('|')])
        j += 1
    return rows, j


def _num(s):
    try:
        return float(s)
    except ValueError:
        return None


def parse_log(path):
    """{epoch: metrics} for every evaluation in one mmengine log."""
    lines = open(path, errors='replace').read().splitlines()
    out, cur = {}, {}
    for i, line in enumerate(lines):
        if HDR.search(line):
            rows, _ = _table(lines, i)
            hdr = rows[0]
            for r in rows[1:]:
                name = r[0]
                vals = dict(zip(hdr[1:], r[1:]))
                if name == 'mAP':
                    cur['mAP3D'] = _num(vals.get('AP@3D', ''))
                    cur['mAPBEV'] = _num(vals.get('AP@BEV', ''))
                    continue
                cur[f'{name}_n_gt'] = _num(vals.get('n_gt', ''))
                cur[f'{name}_AP3D'] = _num(vals.get('AP@3D', ''))
                cur[f'{name}_APBEV'] = _num(vals.get('AP@BEV', ''))
                for k, v in vals.items():
                    if k.startswith('3D '):
                        cur[f'{name}_AP3D_{k[3:]}'] = _num(v)
        elif DIST_HDR.search(line):
            rows, _ = _table(lines, i)
            hdr = rows[0]
            for r in rows[1:]:
                vals = dict(zip(hdr[1:], r[1:]))
                if r[0] == 'mAP':
                    cur['mAPdist'] = _num(vals.get('mean', '') or r[-1])
                else:
                    cur[f'{r[0]}_APdist'] = _num(vals.get('mean', ''))
        else:
            m = EPOCH.search(line)
            if m and cur and 'Maritime/' in line and 'eta' not in line:
                key = int(m.group(2)) if m.group(1) == 'val' else 'test'
                out[key] = cur
                cur = {}
    return out


def find_log(run):
    if os.path.isfile(run):
        return run
    logs = sorted(glob.glob(os.path.join(run, '*', '*.log')),
                  key=os.path.getmtime)
    return logs[-1] if logs else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--json')
    ap.add_argument('--cols', default='mAP3D,mAPBEV,mAPdist,boat_AP3D,'
                    'boat_APBEV,buoy_AP3D,sailboat_AP3D,ship_AP3D,yacht_AP3D')
    args = ap.parse_args()
    cols = args.cols.split(',')
    allres = {}
    for run in args.runs:
        log = find_log(run)
        if not log:
            continue
        hist = parse_log(log)
        name = os.path.basename(os.path.normpath(run))
        allres[name] = hist
        print(f'== {name}')
        print('   epoch ' + ''.join(f'{c:>14s}' for c in cols))
        for ep in sorted(k for k in hist if k != 'test'):
            v = hist[ep]
            print(f'   {ep:5d} ' + ''.join(
                f'{v[c]:14.2f}' if v.get(c) is not None else f'{"-":>14s}'
                for c in cols))
    if args.json:
        json.dump(allres, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
