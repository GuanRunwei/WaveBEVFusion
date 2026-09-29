"""Shared dataset/eval settings for the maritime3d benchmark (sequence 00).

Detection range covers +-160 m because annotated vessels reach 178 m -- far
beyond the 70 m KITTI setting. Frames without ground truth are kept in the
evaluation splits so that false positives on empty water are penalised.
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d'], allow_failed_imports=False)

dataset_type = 'MaritimeDataset'
data_root = 'data/maritime/'
class_names = ['boat', 'ship', 'sailboat', 'buoy']
metainfo = dict(classes=class_names)

# x/y cover the full annotated extent; z spans hull bottoms to mast tops
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'maritime_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(boat=10, ship=10, sailboat=10, buoy=5)),
    classes=class_names,
    # rare classes are oversampled hardest: ship is 130x rarer than boat
    sample_groups=dict(boat=4, ship=8, sailboat=6, buoy=6),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    backend_args=backend_args)

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.5, 0.5, 0.2]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=point_cloud_range)
        ]),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        # one pass per epoch; 40 epochs over the 6586 non-empty training frames
        # is ~263k samples seen, the same budget KITTI baselines get from
        # 80 epochs over 3712 frames
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            # empty frames are dropped for training but kept for evaluation
            ann_file='maritime_infos_train_nonempty.pkl',
            data_prefix=dict(pts='points'),
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            backend_args=backend_args)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='maritime_infos_val.pkl',
        data_prefix=dict(pts='points'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        backend_args=backend_args))

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='maritime_infos_test.pkl',
        data_prefix=dict(pts='points'),
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        filter_empty_gt=False,
        backend_args=backend_args))

val_evaluator = dict(
    type='MaritimeMetric', pcd_limit_range=point_cloud_range)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
