"""TransFusion-L + WAHead (VWA kappa=0.5 + SSR) on the maritime3d benchmark.

Identical to transfusion_lidar_maritime-bench.py except for the head, so the
difference to that run is the WAHead contribution (M2). Ablations:
  --cfg-options model.bbox_head.bbox_coder.kappa=0      -> SSR only
  --cfg-options model.bbox_head.ssr=False \
      model.bbox_head.common_heads.fb=None             -> VWA only
"""
_base_ = ['./transfusion_lidar_maritime-bench.py']

custom_imports = dict(
    imports=[
        'projects.Maritime3D.maritime3d', 'projects.BEVFusion.bevfusion',
        'projects.Maritime3D.maritime3d.wahead'
    ],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='WAHead',
        ssr=True,
        plane_scale=100.0,
        plane_prior_lambda=(1.0, 1.0, 1.0),
        plane_prior_c=-1.3,
        loss_plane_weight=0.5,
        loss_height_raw_weight=0.25,
        loss_center_weight=0.25,
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], fb=[1, 2]),
        bbox_coder=dict(type='WaterlineBBoxCoder', kappa=0.5)))
