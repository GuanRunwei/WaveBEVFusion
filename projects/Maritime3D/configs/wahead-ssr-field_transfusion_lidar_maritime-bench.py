"""Physical sea model for SSR (``sea_field``): IMU tilt + Kalman-filtered
mean level + Gaussian-process wave field averaged over each hull, fused
with per-query learned uncertainties (Laplace NLL). See WAHead's docstring.

Motivation (train, 199k boxes, IMU tilt): box bottoms deviate from the
frame's shared level by 0.36 m median (p90 1.36 m), and the deviations of
two hulls are correlated +0.52 within 20 m, +0.30 at 20-50 m and ~0 beyond
50 m -- local waves, not one rigid plane. With the rigid plane
(bench_gn_ssr_imu_r2 ep10) a query's own bottom was closer to the GT than
the SSR output in 55% of cases.
Compare with wahead-ssr-imu-temporal_transfusion_lidar_maritime-bench.py.
The Kalman prediction is a random walk (heave_gain=0): the IMU heave is real
vessel motion (6.6-8.7 cm RMS) but explains <1% of the frame-to-frame change
of the annotated sea level. Training simulates the prior (option A).
"""
_base_ = ['./wahead-ssr-imu-temporal_transfusion_lidar_maritime-bench.py']

model = dict(
    bbox_head=dict(
        sea_field=True,
        # annotations follow the hull's pitch / roll only partly
        tilt_gain=True,
        field_obs=50,
        wave_sigma=0.3,
        wave_length=25.0,
        level_prior_var=1.0,
        hist_std_range=(0.05, 1.0),
        # measured on train+val (work_dirs/tmp/horizon/heave/report.txt):
        # the annotated level drifts ~4.9e-3 m^2 per 0.1 s, and IMU heave
        # explains <1% of it (wrong sign) -> prediction is a random walk
        kalman_q=0.005,
        heave_gain=0.0,
        loss_sigma_weight=0.25,
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2], fb=[1, 2],
            sig=[1, 2])))


# heave for the test-time Kalman prediction; a missing file only marks every
# frame's heave invalid (random-walk prediction). Training never uses it.
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
    dict(type='LoadSeaUp', lookup=sea_up_lookup,
         heave_lookup=sea_heave_lookup),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'], meta_keys=meta_keys)
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline))
