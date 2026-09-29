"""SECOND (sparse-conv VoxelNet) baseline for the maritime3d benchmark.

Shares the anchor sizes, assigners and schedule with the PointPillars baseline
so the two are directly comparable; the difference is the 3D sparse-conv
backbone in place of pillars. Voxel z-resolution is 0.4 m over the -8..24 m
span, giving 80 z-slices for the sparse encoder to reduce.
"""
_base_ = ['./_base_/maritime-3d-4class.py']

point_cloud_range = {{_base_.point_cloud_range}}
class_names = {{_base_.class_names}}
voxel_size = [0.3125, 0.3125, 0.4]
# (z, y, x) order, as expected by SparseEncoder. 320 m / 0.3125 = 1024 keeps
# the 8x-downsampled BEV grid even, which SECONDFPN's stride-2 branch needs
sparse_shape = [80, 1024, 1024]

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(80000, 100000))),
    voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act')),
    backbone=dict(
        type='SECOND',
        # the sparse encoder collapses the 80-deep z grid to 4 slices of 128
        # channels; KITTI's shallower 41-deep grid leaves only 2 (=256)
        in_channels=512,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4,
        in_channels=512,
        feat_channels=512,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            # per-class sizes are training-set medians (dx, dy, dz); z is the
            # mean box-bottom height for that class
            ranges=[
                [-160.0, -160.0, -1.25, 160.0, 160.0, -1.25],
                [-160.0, -160.0, -2.12, 160.0, 160.0, -2.12],
                [-160.0, -160.0, -2.14, 160.0, 160.0, -2.14],
                [-160.0, -160.0, -2.89, 160.0, 160.0, -2.89],
            ],
            sizes=[
                [17.1, 4.3, 6.1],
                [76.7, 11.0, 20.1],
                [14.6, 4.0, 14.0],
                [2.5, 2.6, 6.5],
            ],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    train_cfg=dict(
        assigner=[
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            # buoys are small; a looser threshold keeps them from starving
            dict(
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.25,
                min_pos_iou=0.25,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.25,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=1000,
        max_num=200))

lr = 0.001
epoch_num = 40
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.4,
        eta_min=lr * 10,
        begin=0,
        end=epoch_num * 0.4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=epoch_num * 0.6,
        eta_min=lr * 1e-4,
        begin=epoch_num * 0.4,
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

default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
