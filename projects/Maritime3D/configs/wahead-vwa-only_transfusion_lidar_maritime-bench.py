"""Ablation: WAHead with VWA (kappa=0.5) only, SSR off.

The freeboard head is removed rather than left idle, since an unused head
would trip DDP's unused-parameter check.
"""
_base_ = ['./wahead_transfusion_lidar_maritime-bench.py']
model = dict(
    bbox_head=dict(
        ssr=False,
        common_heads=dict(
            _delete_=True,
            center=[2, 2],
            height=[1, 2],
            dim=[3, 2],
            rot=[2, 2])))
