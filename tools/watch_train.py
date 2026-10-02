#!/usr/bin/env python3
"""Live view of mmengine training runs: progress, ETA, loss, val metrics, GPUs.

Standard library only, so any python3 works (no numpy / torch needed), on any
machine with mmengine-style logs (<work_dir>/<timestamp>/<timestamp>.log).

    python tools/watch_train.py                        # newest runs in work_dirs/
    python tools/watch_train.py work_dirs/bench_cp     # one or more work dirs / .log files
    python tools/watch_train.py work_dirs/a work_dirs/b -n 60
    python tools/watch_train.py work_dirs/a --keys '3D_mAP|BEV'   # pick val columns
    python tools/watch_train.py work_dirs/a --once     # print once and exit

Ctrl+C to quit.
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta

TS = re.compile(r'^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})')
TRAIN = re.compile(r'(Epoch|Iter)\(train\)\s+(?:\[(\d+)\])?\[\s*(\d+)/(\d+)\]')
VAL = re.compile(r'(Epoch|Iter)\(val\)\s+\[(\d+)\](?:\[\s*\d+/\d+\])?')
KV = re.compile(r'(?<![\w/@.\-])([A-Za-z][\w/@.\-]*): (-?\d+\.?\d*(?:e[-+]?\d+)?)')
ETA = re.compile(r'eta: ((?:\d+ days?, )?\d+:\d{2}:\d{2})')
MAX_EP = re.compile(r'max_epochs\s*=\s*(\d+)')
MAX_IT = re.compile(r'max_iters\s*=\s*(\d+)')


def find_log(path):
    if os.path.isfile(path):
        return path
    logs = glob.glob(os.path.join(path, '*', '*.log'))
    return max(logs, key=os.path.getmtime) if logs else None


def newest_runs(root='work_dirs', k=3, hours=48):
    logs = glob.glob(os.path.join(root, '*', '*', '*.log'))
    fresh = [p for p in logs if time.time() - os.path.getmtime(p) < hours * 3600]
    fresh.sort(key=os.path.getmtime, reverse=True)
    seen, out = set(), []
    for p in fresh:
        run = os.path.dirname(os.path.dirname(p))
        if run not in seen:
            seen.add(run)
            out.append(run)
    return out[:k]


def parse_eta(s):
    d = 0
    if 'day' in s:
        dpart, s = s.split(', ')
        d = int(dpart.split()[0])
    h, m, sec = map(int, s.split(':'))
    return timedelta(days=d, hours=h, minutes=m, seconds=sec)


def parse(log):
    max_ep = max_it = None
    last_train, last_ts, vals, errors = None, None, {}, []
    with open(log, errors='replace') as f:
        for line in f:
            if max_ep is None and (m := MAX_EP.search(line)):
                max_ep = int(m.group(1))
            if max_it is None and (m := MAX_IT.search(line)):
                max_it = int(m.group(1))
            if (m := TS.match(line)):
                last_ts = m.group(1)
            if 'Traceback' in line or re.search(r'\bloss: nan', line):
                errors.append(line.strip()[:160])
            if (m := TRAIN.search(line)):
                last_train = (m, line)
                continue
            if (m := VAL.search(line)) and 'eta:' not in line:
                kv = dict(KV.findall(line[m.end():]))
                if kv:  # metric summary line (progress lines carry eta)
                    vals.setdefault(int(m.group(2)), {}).update(
                        {k: float(v) for k, v in kv.items()})
    return dict(max_ep=max_ep, max_it=max_it, train=last_train,
                last_ts=last_ts, vals=vals, errors=errors)


def pick_keys(vals, pattern, ncol):
    keys = []
    for v in vals.values():
        keys += [k for k in v if k not in keys]
    if pattern:
        return [k for k in keys if re.search(pattern, k)][:ncol]
    # default: the shortest *mAP* keys are the overall numbers
    cand = [k for k in keys if 'mAP' in k] or keys
    short = sorted(cand, key=len)[:ncol]
    return [k for k in keys if k in short]


def fmt_val(x):
    return f'{x * 100:.2f}' if 0 <= x <= 1 else f'{x:.3g}'


def show(run, args):
    log = find_log(run)
    name = os.path.basename(os.path.normpath(run))
    if log is None:
        return f'== {name}: 没找到日志\n'
    r = parse(log)
    # the maritime metric's own tables are per evaluation; the logged
    # summary of runs before 2026-09-29 is a running mean (see
    # tools/maritime_val_history.py), so prefer the tables
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from maritime_val_history import parse_log as parse_tables
        for ep, v in parse_tables(log).items():
            if ep in r['vals']:
                r['vals'][ep].update({f'Maritime/{k}': x / 100.0
                                      for k, x in v.items()
                                      if x is not None and 'n_gt' not in k})
    except Exception:
        pass
    out = [f'== {name}   ({log})']
    now = datetime.now()
    if r['last_ts']:
        age = now - datetime.strptime(r['last_ts'], '%Y/%m/%d %H:%M:%S')
        stale = '   ⚠ 超过 10 分钟没有新日志，进程可能已停止' \
            if age > timedelta(minutes=10) else ''
        out.append(f'   最后更新 {r["last_ts"]}（{int(age.total_seconds() // 60)} 分钟前）{stale}')
    if r['train']:
        m, line = r['train']
        kv = dict(KV.findall(line[m.end():]))
        by_epoch = m.group(1) == 'Epoch'
        it, n_it = int(m.group(3)), int(m.group(4))
        if by_epoch:
            ep = int(m.group(2))
            done = ((ep - 1) * n_it + it) / (r['max_ep'] * n_it) \
                if r['max_ep'] else None
            pos = f'epoch {ep}/{r["max_ep"] or "?"}  iter {it}/{n_it}'
        else:
            done = it / n_it
            pos = f'iter {it}/{n_it}'
        bar = ''
        if done is not None:
            k = int(done * 30)
            bar = f'[{"#" * k}{"." * (30 - k)}] {done * 100:5.1f}%  '
        out.append(f'   {bar}{pos}')
        stopped = r['last_ts'] and now - datetime.strptime(
            r['last_ts'], '%Y/%m/%d %H:%M:%S') > timedelta(minutes=10)
        if (e := ETA.search(line)) and not stopped:
            fin = now + parse_eta(e.group(1))
            out.append(f'   剩余 {e.group(1)}  → 预计 {fin:%m-%d %H:%M} 结束')
        losses = {k: v for k, v in kv.items() if 'loss' in k}
        main = [f'{k} {kv[k]}' for k in ('lr', 'time', 'data_time', 'memory',
                                         'grad_norm') if k in kv]
        out.append('   ' + '  '.join(main))
        shown = ['loss'] + ([k for k in losses if k != 'loss']
                            if args.all_losses else [])
        extra = [k for k in kv if 'loss' not in k and k not in
                 ('lr', 'eta', 'time', 'data_time', 'memory', 'grad_norm')]
        out.append('   ' + '  '.join(f'{k} {kv[k]}' for k in shown + extra
                                     if k in kv))
    if r['vals']:
        keys = pick_keys(r['vals'], args.keys, args.ncol)
        w = max(8, *(len(k.split('/')[-1]) for k in keys))
        out.append('   val:  ' + 'epoch'.ljust(7) +
                   ''.join(k.split('/')[-1].rjust(w + 2) for k in keys))
        best = max(r['vals'].items(),
                   key=lambda kv: kv[1].get(keys[0], float('-inf'))) \
            if keys else None
        for ep in sorted(r['vals']):
            v = r['vals'][ep]
            star = ' ⭐' if best and ep == best[0] else ''
            out.append('         ' + str(ep).ljust(7) + ''.join(
                (fmt_val(v[k]) if k in v else '-').rjust(w + 2)
                for k in keys) + star)
    else:
        out.append('   val:  还没有评测结果')
    for e in r['errors'][-2:]:
        out.append(f'   !! {e}')
    return '\n'.join(out) + '\n'


def gpus():
    try:
        q = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,'
             'memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ''
    cells = []
    for row in q.splitlines():
        i, u, m, t = [x.strip() for x in row.split(',')]
        cells.append(f'{i}:{u:>3}% {int(m) / 1024:4.1f}/{int(t) / 1024:.0f}G')
    return '== GPU  ' + '  '.join(cells) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('runs', nargs='*', help='work dirs or .log files')
    ap.add_argument('-n', '--interval', type=float, default=30)
    ap.add_argument('--keys', help='regex of val metrics to show')
    ap.add_argument('--ncol', type=int, default=6)
    ap.add_argument('--all-losses', action='store_true')
    ap.add_argument('--root', default='work_dirs',
                    help='where to look when no run is given')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    while True:
        runs = args.runs or newest_runs(args.root)
        text = f'{datetime.now():%Y-%m-%d %H:%M:%S}\n' + gpus() + '\n' + \
            '\n'.join(show(r, args) for r in runs)
        if args.once:
            print(text)
            return
        sys.stdout.write('\033[2J\033[H' + text +
                         f'\n（每 {args.interval:g} 秒刷新，Ctrl+C 退出）\n')
        sys.stdout.flush()
        time.sleep(args.interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
