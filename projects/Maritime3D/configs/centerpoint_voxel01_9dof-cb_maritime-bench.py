"""CenterPoint + target attitude (pitch, roll) + class-balanced heatmap loss.

Identical to centerpoint_voxel01_maritime-bench.py (same data, augmentation,
schedule and metric) except:
- 9-DoF boxes [x, y, z, l, w, h, yaw, pitch, roll] (MaritimePoseBoxes, so
  the rotate / scale / flip augmentations keep pitch and roll consistent),
  with (sin, cos) pitch and roll heads; MaritimeMetric then also reports
  AOE_pitch / AOE_roll;
- the effective-number class-balanced Gaussian focal loss of the other
  workstation's long-tail recipe (beta 0.9999 on the v5 train box counts ->
  weights boat 0.86, buoy 3.11, sailboat 1.40, ship 5.05, yacht 4.09).
  Its repeat-factor sampler is left out: at repeat_thr 0.05 it barely
  touches this split (ship frames x1.15 with 1/3 of empty frames kept).
"""
_base_ = ['./centerpoint_voxel01_maritime-bench.py']

# v5 train boxes per class, label order boat, buoy, sailboat, ship, yacht
class_freq = [189141, 3251, 9591, 1875, 2374]

model = dict(
    pts_bbox_head=dict(
        type='MaritimeCenterHead',
        common_heads=dict(pitch=(2, 2), roll=(2, 2)),
        bbox_coder=dict(type='MaritimePoseCenterPointBBoxCoder', code_size=9),
        loss_cls=dict(
            _delete_=True,
            type='MaritimeClassBalancedGaussianFocalLoss',
            num_classes=5,
            class_freq=class_freq,
            beta=0.9999,
            gamma=4.0,
            alpha=2.0,
            bg_weight=1.0,
            loss_weight=1.0)),
    train_cfg=dict(pts=dict(code_weights=[1.0] * 12)))

pose = dict(box_type_3d='Maritime', box_3d_dof=9)
train_dataloader = dict(dataset=pose)
val_dataloader = dict(dataset=pose)
test_dataloader = dict(dataset=pose)
