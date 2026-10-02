"""BEVFusion-style camera fusion on CenterPoint: the baseline for MariFusion
with the same LiDAR detector, start point and schedule.

Camera branch, view transform and fuser are those of
bevfusion_lidar-cam_maritime-bench.py (Swin-T + GeneralizedLSSFPN, 0.5x
horizon band 1024x320 per camera, DepthLSSTransform to a 400x400 BEV,
ConvFuser before the 2D backbone); the LiDAR path and the CenterHead are
gn_centerpoint_voxel01_maritime-bench.py's. Initialised from bench_gn_cp
(20 ep) and trained 6 more epochs, as gn_centerpoint-marifusion.
"""
_base_ = ['./gn_centerpoint_voxel01_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=True)
image_size = [320, 1024]
data_prefix = dict(pts='', CAM_FRONT_LEFT='', CAM_FRONT_RIGHT='')

model = dict(
    type='BEVFuseCenterPoint',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://github.com/SwinTransformer/storage/releases/'
            'download/v1.0.0/swin_tiny_patch4_window7_224.pth')),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=256,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    view_transform=dict(
        type='DepthLSSTransform',
        in_channels=256,
        out_channels=80,
        image_size=image_size,
        feature_size=[image_size[0] // 8, image_size[1] // 8],
        xbound=[-160.0, 160.0, 0.4],
        ybound=[-160.0, 160.0, 0.4],
        zbound=[-8.0, 24.0, 32.0],
        dbound=[1.0, 160.0, 1.0],
        downsample=2),
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 512], out_channels=512))

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
        resize_lim=[0.48, 0.52],
        bot_pct_lim=[0.15, 0.19],
        rot_lim=[-2.0, 2.0],
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
        resize_lim=[0.5, 0.5],
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

# 6 epochs from the LiDAR checkpoint, as gn_centerpoint-marifusion
load_from = 'work_dirs/bench_gn_cp/epoch_20.pth'
epoch_num = 6
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=3)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epoch_num, end=epoch_num,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
]
