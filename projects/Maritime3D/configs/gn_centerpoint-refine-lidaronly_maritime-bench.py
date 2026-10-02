"""Control for MariFusion: the same second stage (RaySplitFusionRefineHead)
and schedule without any camera input, so that
gn_centerpoint-marifusion minus this run is the cameras' contribution and
this run minus bench_gn_cp is the refinement's.
"""
_base_ = ['./gn_centerpoint-marifusion_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
model = dict(
    img_backbone=None,
    img_neck=None,
    refine_head=dict(use_image=False))
# images are not needed; the LiDAR part of the pipeline is unchanged
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
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys={{_base_.meta_keys}})
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Pack3DDetInputs', keys=['points'],
         meta_keys={{_base_.meta_keys}})
]
_ds = dict(modality=dict(use_lidar=True, use_camera=False))
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, **_ds))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))
