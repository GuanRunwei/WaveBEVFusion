"""Shared dataset/eval settings for the maritime3d benchmark on the official
splits (dataset/splits/detection, 282/60/59 clips).

Infos come from tools/create_maritime_infos_from_tables.py:
- dataset/infos/detection/maritime_infos_{train,val,test}.pkl
- boxes [x, y, z_bottom, l, w, h, yaw, pitch, roll] in the LiDAR frame
- verified calibration (xyzw, sensor->body)
- compact (N, 4) points from tools/compact_maritime_points.py, so
  load_dim=use_dim=4 and no padding strip is needed.

Training keeps every annotated frame plus one in three open-water frames
(`empty_frame_keep`). Evaluation keeps every frame, so false positives on
empty water still count.
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d'], allow_failed_imports=False)

dataset_type = 'MaritimeDataset'
data_root = 'dataset/'
info_prefix = 'infos/detection/'
class_names = ['boat', 'buoy', 'sailboat', 'ship', 'yacht']
metainfo = dict(
    classes=class_names,
    palette=[(0, 120, 255), (255, 210, 0), (0, 200, 120), (255, 90, 0),
             (200, 0, 255)])

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'])
]


common = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=dict(pts=''),
    modality=input_modality,
    metainfo=metainfo,
    box_type_3d='LiDAR',
    filter_empty_gt=False,
    backend_args=backend_args)

train_dataloader = dict(
    batch_size=4,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        **common,
        ann_file=info_prefix + 'maritime_infos_train.pkl',
        pipeline=train_pipeline,
        test_mode=False,
        empty_frame_keep=1.0 / 3))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **common,
        ann_file=info_prefix + 'maritime_infos_val.pkl',
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        **common,
        ann_file=info_prefix + 'maritime_infos_test.pkl',
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(
    type='MaritimeMetric',
    pcd_limit_range=point_cloud_range,
    iou_thresholds=dict(
        boat=0.5, buoy=0.25, sailboat=0.5, ship=0.5, yacht=0.5))
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
