"""Final model on CenterFormer, stage 2: MariFusion on CenterFormer + sea
surface (gn_centerpoint-sea-marifusion_maritime-bench.py is the CenterPoint
version).

Stage 1 is gn_centerformer-sea_maritime-bench-80e.py (80 epochs); this
config adds the two front cameras (full-resolution horizon band, ResNet-50
C3 / C4 + FPN) and the RaySplitFusionRefineHead, which samples
CenterFormer's 800 x 800 BEV map (256 channels) and the images for each
stage-1 detection, and trains 6 more epochs from that checkpoint. Same
pipelines, schedule and image-backbone lr as the CenterPoint version.
"""
_base_ = ['./gn_centerformer-sea_maritime-bench-80e.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=True)
image_size = [640, 2048]
data_prefix = dict(pts='', CAM_FRONT_LEFT='', CAM_FRONT_RIGHT='')
sea_up_lookup = 'dataset/infos/detection/sea_up_lidar.npz'
sea_heave_lookup = 'dataset/infos/detection/sea_heave_lidar.npz'

model = dict(
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
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
        bev_channels=256,
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

# MariFusion's keys + LoadSeaUp's
meta_keys = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
    'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
    'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats',
    'sea_up', 'sea_up_valid', 'sea_heave', 'sea_heave_valid', 'sea_up_rot'
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
    dict(type='EvidenceAnchor', mode='centre'),
    dict(type='LoadSeaUp', lookup=sea_up_lookup),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsAnchor',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
              'gt_anchors_3d'],
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
    dict(type='LoadSeaUp', lookup=sea_up_lookup,
         heave_lookup=sea_heave_lookup),
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

load_from = 'work_dirs/bench_gn_cf_sea_80e/epoch_80.pth'
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
