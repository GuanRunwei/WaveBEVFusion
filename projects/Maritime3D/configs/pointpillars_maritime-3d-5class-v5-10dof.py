"""PointPillars baseline with full 10-DoF regression on the v5 benchmark.

Identical to pointpillars_maritime-3d-5class-v5.py except that the head
additionally regresses target pitch and roll (annotation columns 7/8), so
the benchmark's 10-DoF annotation is supervised end to end and scored:

- anchors carry two extra custom dims, pinned at 0 (pitch/roll of a level
  hull), giving 9-wide anchors via the stock AlignedAnchor3DRangeGenerator;
- DeltaXYZWLHRBBoxCoder(code_size=9) encodes pitch/roll as raw residuals
  against those zero anchors -- i.e. the head regresses them directly,
  while x/y/z/w/l/h/yaw keep the standard delta parameterisation;
- the dataset exposes 9-wide ground truth (box_3d_dof=9) so the assigner,
  targets and eval annotations all carry pitch/roll;
- MaritimeMetric reports AOE_pitch / AOE_roll over the same
  centre-distance-matched pairs as yaw AOE.

Direction classification, losses and NMS are unchanged: yaw keeps its
sin-difference encoding and flip classifier, and pitch/roll ride the same
SmoothL1 term with the existing loss weights.
"""
_base_ = ['./pointpillars_maritime-3d-5class-v5.py']

# Longer schedule than the 7-DoF run: the two extra regression dims give
# the head more to fit, and the near-zero pitch/roll targets only get
# pinned late in training when the lr is low, hence 60 epochs (vs 40).
# The scheduler restates the base's -- with the same correction as the
# base: 500-iter linear warmup then ONE cosine decay to lr*1e-4. The
# upstream two-phase eta_min=lr*10 form made lr rise to 10x base; see
# the comment in pointpillars_maritime-3d-5class-v5.py for the measured
# damage that caused.
epoch_num = 60
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
lr = 0.001
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        begin=0,
        end=500,
        by_epoch=False),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num,
        eta_min=lr * 1e-4,
        begin=0,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=epoch_num * 0.6,
        eta_min=1,
        begin=epoch_num * 0.4,
        end=epoch_num,
        by_epoch=True,
        convert_to_iter_based=True),
]

model = dict(
    bbox_head=dict(
        anchor_generator=dict(custom_values=[0, 0]),
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder', code_size=9)))

train_dataloader = dict(
    dataset=dict(dataset=dict(box_3d_dof=9)))
val_dataloader = dict(
    dataset=dict(box_3d_dof=9))
test_dataloader = dict(
    dataset=dict(box_3d_dof=9))
