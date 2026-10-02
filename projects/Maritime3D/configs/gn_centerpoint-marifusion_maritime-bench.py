"""MariFusion: CenterPoint (GroupNorm) + two front cameras, two-stage sparse
instance fusion (RaySplitFusionRefineHead): each detection samples the
full-resolution horizon band of both cameras on a reflection-aware grid over
its projection, a per-camera reliability gate (valid fraction, brightness,
range; modality dropout) weights the image evidence, and the residual along
the viewing ray / in height comes from LiDAR only while bearing, size,
heading and score use the fused feature.

Plain CenterHead (no evidence anchor, no sea surface) so the fusion's
contribution is measured alone. Initialised from bench_gn_cp (20 ep) and
trained 6 more epochs; the control without cameras is
gn_centerpoint-refine-lidaronly_maritime-bench.py (same schedule).
"""
_base_ = ['./gn_centerpoint_voxel01_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=True)
# full-resolution horizon band: rows ~256-896 of the 2048 x 1080 images
image_size = [640, 2048]
data_prefix = dict(pts='', CAM_FRONT_LEFT='', CAM_FRONT_RIGHT='')

model = dict(
    type='MariFusionCenterPoint',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        # C3, C4 (strides 8, 16); stage 4 is not built, so every parameter
        # gets a gradient (DDP)
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(1, 2),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024],
        out_channels=256,
        num_outs=2),
    refine_head=dict(
        type='RaySplitFusionRefineHead',
        num_classes=5,
        bev_channels=512,
        img_channels=256,
        hidden=256,
        pc_range=point_cloud_range,
        use_image=True,
        num_levels=2,
        grid=4,
        widen=0.2,
        waterline_margin=2.0,
        drop_all=0.25,
        drop_single=0.1,
        pos_iou=0.25,
        max_train_props=128,
        gt_jitter=True,
        score_alpha=0.5))

meta_keys = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
    'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
    'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats'
]
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=None),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.15, 0.19],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
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
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=None),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.17, 0.17],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs', keys=['img', 'points'], meta_keys=meta_keys)
]
_ds = dict(
    modality=input_modality,
    data_prefix=data_prefix,
    default_cam_key='CAM_FRONT_LEFT')
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, **_ds))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))

# 6 epochs from the LiDAR checkpoint (as BEVFusion-LC); image backbone at
# 0.1x lr
load_from = 'work_dirs/bench_gn_cp/epoch_20.pth'
epoch_num = 6
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=3)
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epoch_num, end=epoch_num,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
]
