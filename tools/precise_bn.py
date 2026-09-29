"""Re-estimate BatchNorm statistics of a trained checkpoint (Precise BN).

With 2 frames per GPU and scenes that range from open water to crowded
harbours, a BN layer's batch statistics are noisy, and its running
statistics (momentum 0.01, i.e. an average over roughly the last hundred
iterations) end up off by more than one standard deviation in some
checkpoints (e.g. pts_backbone.blocks.1.* in bench_tfl_v2/epoch_20 and
bench_tfl_ssr_dt/epoch_10), which shrinks the predicted boxes and scrambles
their yaw in eval mode only. This replaces every BN layer's running
statistics with a plain average over ``--iters`` training batches, run in
train mode without gradients; the weights are untouched.

usage: python tools/precise_bn.py <config> <ckpt> <out_ckpt> [--iters 200]
"""
import argparse
import copy
import importlib

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint, save_checkpoint

from mmdet3d.registry import DATASETS, MODELS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('ckpt')
    ap.add_argument('out')
    ap.add_argument('--iters', type=int, default=200)
    ap.add_argument('--batch', type=int, default=2)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    init_default_scope('mmdet3d')
    cfg = Config.fromfile(args.config)
    for m in cfg.get('custom_imports', {}).get('imports', []):
        importlib.import_module(m)
    ds = DATASETS.build(copy.deepcopy(cfg.train_dataloader.dataset))
    model = MODELS.build(copy.deepcopy(cfg.model))
    ckpt = load_checkpoint(model, args.ckpt, map_location='cpu')
    model = model.cuda().train()

    bns = [m for m in model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None  # cumulative moving average
    g = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(ds), generator=g).tolist()
    with torch.no_grad():
        for it in range(args.iters):
            idx = order[it * args.batch:(it + 1) * args.batch]
            data = model.data_preprocessor(
                pseudo_collate([ds[i] for i in idx]), True)
            model(**data, mode='loss')
    for bn in bns:
        bn.momentum = 0.01
    meta = dict(ckpt.get('meta', {}))
    meta['precise_bn'] = dict(source=args.ckpt, iters=args.iters,
                              batch=args.batch, seed=args.seed)
    save_checkpoint(dict(state_dict=model.state_dict(), meta=meta), args.out)
    print(f'{len(bns)} BN layers re-estimated over {args.iters} x '
          f'{args.batch} training frames -> {args.out}')


if __name__ == '__main__':
    main()
