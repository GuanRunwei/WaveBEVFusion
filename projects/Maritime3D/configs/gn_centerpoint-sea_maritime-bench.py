"""Ablation: the physical sea surface on plain CenterPoint (heatmap at the
box centre; the anchor-to-centre vector is then zero).
Compare with gn_centerpoint-ea-sea_maritime-bench.py.
"""
_base_ = ['./gn_centerpoint-ea-sea_maritime-bench.py']

sea_up_lookup = 'dataset/infos/detection/sea_up_lidar.npz'
sea_heave_lookup = 'dataset/infos/detection/sea_heave_lidar.npz'
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
# Pack3DDetInputs defaults + LoadSeaUp's keys (SEA_HEAVE_META_KEYS)
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
    'sea_heave',
    'sea_heave_valid',
    'sea_up_rot',
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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='EvidenceAnchor', mode='centre'),
    dict(type='LoadSeaUp', lookup=sea_up_lookup),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputsAnchor',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_anchors_3d'],
        meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadSeaUp', lookup=sea_up_lookup,
         heave_lookup=sea_heave_lookup),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=meta_keys)
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
