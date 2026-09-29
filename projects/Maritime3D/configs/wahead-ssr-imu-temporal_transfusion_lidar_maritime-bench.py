"""Temporal, decoupled SSR with the sea tilt taken from the IMU.

The LiDAR is body-fixed, so the sea tilts in its frame with the hull's
attitude, and the tilt of a plane fitted to annotated box bottoms is mostly
label noise (~0.7 deg; work_dirs/tmp/horizon). Here (a, b) come from the
per-frame gravity direction (LoadSeaUp, after the augmentation so it follows
the points) plus a learned mounting offset, and only the sea level c is
solved from the queries and carried by the temporal stream.
Compare with wahead-ssr-temporal_transfusion_lidar_maritime-bench.py.
"""
_base_ = ['./wahead-ssr-temporal_transfusion_lidar_maritime-bench.py']

sea_up_lookup = 'dataset/infos/detection/sea_up_lidar.npz'
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]

model = dict(bbox_head=dict(imu_tilt=True))

# Pack3DDetInputs defaults + the two LoadSeaUp adds (SEA_UP_META_KEYS)
meta_keys = (
    'img_path',
    'ori_shape',
    'img_shape',
    'lidar2img',
    'depth2img',
    'cam2img',
    'pad_shape',
    'scale_factor',
    'flip',
    'pcd_horizontal_flip',
    'pcd_vertical_flip',
    'box_mode_3d',
    'box_type_3d',
    'img_norm_cfg',
    'num_pts_feats',
    'pcd_trans',
    'sample_idx',
    'pcd_scale_factor',
    'pcd_rotation',
    'pcd_rotation_angle',
    'lidar_path',
    'transformation_3d_flow',
    'trans_mat',
    'affine_aug',
    'sweep_img_metas',
    'ori_cam2img',
    'cam2global',
    'crop_offset',
    'img_crop_offset',
    'resize_img_shape',
    'lidar2cam',
    'ori_lidar2img',
    'num_ref_frames',
    'num_views',
    'ego2global',
    'axis_align_matrix',
    'sea_up',
    'sea_up_valid',
)

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
    dict(type='LoadSeaUp', lookup=sea_up_lookup),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadSeaUp', lookup=sea_up_lookup),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=meta_keys)
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
