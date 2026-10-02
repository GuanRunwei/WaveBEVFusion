"""TransFusion-L on the maritime3d benchmark (official splits, 5 classes).

LiDAR-only baseline (M1) and the base the WAHead variant (M2) builds on.
- range +-160 m, voxel 0.1 m (xy) x 0.4 m (z) -> 3200x3200x80 grid; the 8x
  sparse encoder gives a 400x400 BEV at 0.8 m, fine enough for 2.5 m buoys;
- 20 epochs, warmup + cosine at lr 1e-4, on 8 GPUs x 2 frames. Batch 4 per
  GPU overflows mmcv's int32 sparse-conv indexing (3200*3200*80*4 > 2^31,
  CUDA illegal memory access);
- no GT-paste yet (dbinfos for the new split are not built).
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d', 'projects.BEVFusion.bevfusion'],
    allow_failed_imports=False)

_base_ = ['./_base_/maritime-3d-5class-bench.py']

# spelled out (not {{_base_.x}}) because the head slices it
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
voxel_size = [0.1, 0.1, 0.4]
grid_size = [3200, 3200, 80]
out_size_factor = 8
num_classes = 5

model = dict(
    type='BEVFusion',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxelize_cfg=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=[90000, 120000],
            voxelize_reduce=True)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='BEVFusionSparseEncoder',
        in_channels=4,
        sparse_shape=grid_size,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [1, 1, 0]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=512,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    bbox_head=dict(
        type='TransFusionHead',
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=num_classes,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True)),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128)),
        train_cfg=dict(
            dataset='maritime',
            point_cloud_range=point_cloud_range,
            grid_size=grid_size,
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            gaussian_overlap=0.1,
            min_radius=2,
            pos_weight=-1,
            code_weights=[1.0] * 8,
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='mmdet.FocalLossCost',
                    gamma=2.0,
                    alpha=0.25,
                    weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25))),
        test_cfg=dict(
            dataset='maritime',
            grid_size=grid_size,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            pc_range=point_cloud_range[:2],
            nms_type=None),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-180.0, -180.0, -10.0, 180.0, 180.0, 30.0],
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=8),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25)))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='BEVFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.95, 1.05],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=[0.5, 0.5, 0.2]),
    dict(type='BEVFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
train_dataloader = dict(
    batch_size=2, num_workers=8, dataset=dict(pipeline=train_pipeline))

lr = 0.0001
epoch_num = 20
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# Plain warmup + cosine at a 1e-4 peak (the recipe the legacy 4-class
# TransFusion-L run converged with). BEVFusion's cyclic schedule, which
# ramps to 10x lr, destabilised this model: loss rose from 3.35 (ep 5) to
# 6.08 at the 5e-4 peak (ep 8) and val mAP3D halved (work_dirs/bench_tfl).
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epoch_num, end=epoch_num,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.8947, begin=0,
         end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=epoch_num * 0.4,
         end=epoch_num, by_epoch=True, convert_to_iter_based=True),
]
auto_scale_lr = dict(enable=False, base_batch_size=16)

default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=3,
                    save_best='Maritime/mAP3D', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
# mmengine averages every scalar matching its default mean_pattern
# '.*(loss|time|data_time|grad_norm).*' over the window -- and
# 'Mari*time*/mAP3D' matches, so the logged val metrics of earlier runs were
# running means over all evaluations (tools/maritime_val_history.py recovers
# the true values from the tables). Average only the training scalars.
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
    mean_pattern=r'(^|[._/])loss|^time$|^data_time$|^grad_norm$')
log_level = 'INFO'
load_from = None
resume = False

# TensorBoard next to the json scalars (work_dirs/<run>/<ts>/vis_data)
visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'),
                  dict(type='TensorboardVisBackend')],
    name='visualizer')
