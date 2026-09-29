"""CenterPoint (voxel) on the maritime3d benchmark (official splits, 5 classes).

LiDAR-only baseline next to TransFusion-L. Data, augmentation, schedule
(warmup + cosine at lr 1e-4, 20 epochs, 8 GPUs x 2 frames) and evaluation are
inherited unchanged from transfusion_lidar_maritime-bench.py, so the
difference to that run is the architecture: a dense per-class CenterHead
with rotated NMS instead of 200 Hungarian-matched queries.
- same voxel grid: 0.1 x 0.1 x 0.4 m over +-160 m, z in [-8, 24] m
  -> sparse shape [81, 3200, 3200] (z, y, x), 400x400 BEV at 0.8 m;
- one task per class, so the heads for 2.5 m buoys and 80 m ships do not
  share a heatmap.
"""
_base_ = ['./transfusion_lidar_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
voxel_size = [0.1, 0.1, 0.4]
grid_size = [3200, 3200, 80]
out_size_factor = 8
post_range = [-180.0, -180.0, -10.0, 180.0, 180.0, 30.0]

model = dict(
    _delete_=True,
    type='CenterPoint',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000))),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[81, 3200, 3200],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
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
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['boat']),
            dict(num_class=1, class_names=['buoy']),
            dict(num_class=1, class_names=['sailboat']),
            dict(num_class=1, class_names=['ship']),
            dict(num_class=1, class_names=['yacht']),
        ],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=post_range,
            max_num=500,
            score_threshold=0.01,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            grid_size=grid_size,
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0] * 8)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=post_range,
            pc_range=point_cloud_range[:2],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 1, 4, 12, 4],
            score_threshold=0.01,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=1000,
            post_max_size=100,
            nms_thr=0.2)))
