"""wahead-ssr-temporal_transfusion_lidar_maritime-bench.py with GroupNorm (32 groups) instead of BatchNorm in the 2D
LiDAR backbone (SECOND) and neck (SECONDFPN).

With 2 frames per GPU and scenes from open water to crowded harbours, the
BN statistics of these layers were effectively per-frame during training
but population averages at test time; the gap varied from checkpoint to
checkpoint (running mean off by up to 1.6 sd in pts_backbone.blocks.1.*)
and shrank / mis-rotated boxes in eval mode only, e.g. bench_tfl_ssr_dt
ep15 mAP3D 17.8 with eval BN vs 26.8 with per-frame BN statistics
(progress/20260926_181500_BN-train-eval-gap). GroupNorm normalises each
frame on its own, identically in training and test. The sparse encoder's
BN1d (running stats within ~0.14 sd) is kept.
"""
_base_ = ['./wahead-ssr-temporal_transfusion_lidar_maritime-bench.py']

gn = dict(_delete_=True, type='GN', num_groups=32, eps=1e-3)
model = dict(pts_backbone=dict(norm_cfg=gn), pts_neck=dict(norm_cfg=gn))
