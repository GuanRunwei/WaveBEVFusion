"""Ablation: WAHead with SSR only (kappa=0, i.e. the usual centre heatmap)."""
_base_ = ['./wahead_transfusion_lidar_maritime-bench.py']
model = dict(bbox_head=dict(bbox_coder=dict(kappa=0.0)))
