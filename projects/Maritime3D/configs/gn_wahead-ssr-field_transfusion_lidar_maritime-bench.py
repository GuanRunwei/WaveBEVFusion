"""wahead-ssr-field_transfusion_lidar_maritime-bench.py with GroupNorm in
the 2D LiDAR backbone (SECOND) and neck (SECONDFPN); see
gn_transfusion_lidar_maritime-bench.py for why.
"""
_base_ = ['./wahead-ssr-field_transfusion_lidar_maritime-bench.py']

gn = dict(_delete_=True, type='GN', num_groups=32, eps=1e-3)
model = dict(pts_backbone=dict(norm_cfg=gn), pts_neck=dict(norm_cfg=gn))
