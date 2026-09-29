"""BEVFusion (LiDAR + stereo camera) baseline for the maritime3d benchmark.

Shares the LiDAR branch with the TransFusion-L baseline and adds the two forward
stereo cameras through a depth-LSS view transform. The cameras span only ~59 deg
of azimuth while annotations span the full 360, so the camera branch sharpens
detections ahead of the vessel and the LiDAR branch carries every other bearing;
evaluation stays full-circle.

Two deviations from the nuScenes recipe follow from the sensor setup:
  * depth bins run to 160 m instead of 60 m, since vessels are annotated to
    178 m and the LSS frustum has to reach them;
  * images are 2048x1080 rather than 1600x900, so the resize factor is picked to
    keep the full vertical field of view -- the horizon sits around y=430 and
    cropping toward the bottom would throw away distant targets.
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d', 'projects.BEVFusion.bevfusion'],
    allow_failed_imports=False)

_base_ = ['./transfusion_lidar_maritime-3d-4class.py']

point_cloud_range = {{_base_.point_cloud_range}}
class_names = {{_base_.class_names}}
input_modality = dict(use_lidar=True, use_camera=True)

# 2048x1080 -> resize ~0.238 gives 487x257, from which we crop 480x256; that
# drops a single row of sky and <=7 columns, so the whole FOV survives
image_size = [256, 480]
train_resize_lim = [0.238, 0.250]
test_resize_lim = [0.238, 0.238]

# img_path in the info pkls already carries the left_images/right_images
# prefix, so the per-camera prefix is just the data root
data_prefix = dict(pts='points', CAM_LEFT='', CAM_RIGHT='')

model = dict(
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
        # ConvFuser concatenates the camera and LiDAR BEV maps, so they must
        # agree in size: 320 m / 1.25 = 256 cells, downsampled 2x -> 128x128,
        # matching the LiDAR branch's 1024 grid at out_size_factor 8
        xbound=[-160.0, 160.0, 1.25],
        ybound=[-160.0, 160.0, 1.25],
        zbound=[-8.0, 24.0, 32.0],
        dbound=[1.0, 160.0, 1.5],
        downsample=2),
    # keep the LiDAR branch byte-identical to the TransFusion-L baseline so the
    # comparison isolates the camera contribution
    fusion_layer=dict(
        type='ConvFuser', in_channels=[80, 512], out_channels=512))

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
        resize_lim=train_resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
    dict(type='ObjectSample', db_sampler={{_base_.db_sampler}}),
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
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'img_aug_matrix', 'lidar_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
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
        resize_lim=test_resize_lim,
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img'],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'img_aug_matrix', 'lidar_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'num_pts_feats'
        ])
]

# the Swin backbone plus a 106-bin frustum roughly triples the activation
# footprint of the LiDAR-only model, so halve the batch
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality,
            data_prefix=data_prefix)))
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        data_prefix=data_prefix))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        data_prefix=data_prefix))
