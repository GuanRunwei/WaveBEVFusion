"""Shared dataset/eval settings for the maritime3d-5class v5 benchmark.

v5 replaces the legacy sequence-00-only 4-class split with a 4-sequence,
block-based split (tools/resplit_maritime3d_v5.py):
- train/val/test are contiguous 20 s blocks; every split boundary is
  separated by a 10 s temporal buffer, so no near-duplicate frames leak
  across splits;
- every class keeps a floor of instances in val AND test (long-tail safe);
- det train U det val == tracking trainval and det test == tracking test,
  so one annotation set serves both the detection and the tracking
  benchmark;
- seq03 was captured 14 days after 00-02; maritime_split_test_v5_crossday.json
  selects its test frames for an unseen-day generalization evaluation.

Detection range covers +-160 m because annotated vessels reach 178 m -- far
beyond the 70 m KITTI setting. Frames without ground truth are kept in the
evaluation splits so that false positives on empty water are penalised.

Class order follows the v5 pkls' metainfo (alphabetical), which differs from
the legacy 4-class configs.
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d'], allow_failed_imports=False)

dataset_type = 'MaritimeDataset'
data_root = 'dataset/'
det_prefix = 'new_annotations/detection/'
class_names = ['boat', 'buoy', 'sailboat', 'ship', 'yacht']
metainfo = dict(
    classes=class_names,
    palette=[(0, 120, 255), (255, 210, 0), (0, 200, 120), (255, 90, 0),
             (200, 0, 255)])

# x/y cover the full annotated extent; z spans hull bottoms to mast tops
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=False)
backend_args = None

# NOTE: GT-Aug (db_sampler/ObjectSample) is not used in the v5 configs yet:
# no v5 dbinfo pkl exists. Regenerate it from the v5 train split before
# re-enabling -- sampling boxes from the legacy seq00-only dbinfo would leak
# v5-test shapes into training.

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        # raw bins are fixed 131072x7 float32 buffers; real returns are
        # scattered among all-zero padding rows, so load all 7 dims and
        # strip the padding before anything else touches the cloud
        load_dim=7,
        use_dim=4,
        backend_args=backend_args),
    dict(type='StripEmptyLidarPoints'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
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
        # raw bins are fixed 131072x7 float32 buffers; real returns are
        # scattered among all-zero padding rows, so load all 7 dims and
        # strip the padding before anything else touches the cloud
        load_dim=7,
        use_dim=4,
        backend_args=backend_args),
    dict(type='StripEmptyLidarPoints'),
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
        # raw bins are fixed 131072x7 float32 buffers; real returns are
        # scattered among all-zero padding rows, so load all 7 dims and
        # strip the padding before anything else touches the cloud
        load_dim=7,
        use_dim=4,
        backend_args=backend_args),
    dict(type='StripEmptyLidarPoints'),
    dict(type='Pack3DDetInputs', keys=['points'])
]

train_dataloader = dict(
    batch_size=4,
    num_workers=6,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=det_prefix + 'maritime_nuscenes_infos_train_10dof_v5.pkl',
            data_prefix=dict(pts=''),
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
        ann_file=det_prefix + 'maritime_nuscenes_infos_val_10dof_v5.pkl',
        data_prefix=dict(pts=''),
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
        ann_file=det_prefix + 'maritime_nuscenes_infos_test_10dof_v5.pkl',
        data_prefix=dict(pts=''),
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
