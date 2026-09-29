"""PETR (camera-only) baseline for the maritime3d benchmark.

Stands in for BEVFormer, which mmdetection3d does not ship. PETR is the closest
available query-based multi-view camera detector and shares BEVFormer's premise:
3D queries attending to multi-view image features, no LiDAR at all.

The sensor rig makes this baseline structurally different from the LiDAR ones.
Both cameras look forward with a 58.9 deg horizontal FOV, optical axes at
-5.4 deg azimuth, so together they see roughly the sector [-35, +24] deg out of
the full circle the LiDAR annotates. Two consequences, both handled explicitly:

  * training is restricted to that sector (ObjectAzimuthFilter). Supervising the
    model on vessels behind the boat would only teach it to hallucinate;
  * evaluation is reported twice -- on the full circle, so the number sits in
    the same table as every other baseline, and on the camera sector, which is
    the only number that says anything about the method rather than the rig.
    Read the full-circle mAP as an upper bound on what a forward-facing
    camera-only system can do here, not as a verdict on PETR.

Images are 2048x1080; the resize factor keeps the whole horizontal FOV (a
horizontal crop would throw away azimuth the model is being asked about) and
trims only sky from the top.
"""
custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d', 'projects.PETR.petr'],
    allow_failed_imports=False)

_base_ = ['./_base_/maritime-3d-4class.py']

# NOTE: spelled out rather than pulled from _base_ because mmengine substitutes
# {{_base_.x}} textually, and this config slices it. Keep in sync with the base.
point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
class_names = {{_base_.class_names}}
# PETR normalises 3D positions by this range, so it has to be a little wider
# than the annotated volume
position_range = [-180.0, -180.0, -10.0, 180.0, 180.0, 30.0]

# union of the two camera frusta, in degrees from the vessel heading
camera_azimuth_range = [-35.0, 24.0]

input_modality = dict(use_lidar=False, use_camera=True)
# img_path in the info pkls already carries the left_images/right_images prefix
data_prefix = dict(pts='points', CAM_LEFT='', CAM_RIGHT='')

# 2048 * 0.390625 = 800 exactly, so no horizontal crop at test time; 1080 * that
# is 421, of which the top 5 rows (sky) are dropped to reach a 416 height
ida_aug_conf = {
    'resize_lim': (0.390625, 0.43),
    'final_dim': (416, 800),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 1080,
    'W': 2048,
    'rand_flip': True,
}

model = dict(
    type='PETR',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[103.530, 116.280, 123.675],
        std=[57.375, 57.120, 58.395],
        bgr_to_rgb=False,
        pad_size_divisor=32),
    use_grid_mask=True,
    # ResNet-50 rather than the paper's V2-99: the V2-99 weights PETR uses are a
    # DD3D depth-pretrained checkpoint that is not redistributed with mmdet3d,
    # and a benchmark baseline should be reproducible from public weights alone
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained', checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='CPFPN', in_channels=[1024, 2048], out_channels=256, num_outs=2),
    pts_bbox_head=dict(
        type='PETRHead',
        num_classes=4,
        in_channels=256,
        # far fewer objects per frame than nuScenes' 10-class street scenes:
        # a mean of 1.95 ground-truth boxes fall inside the camera sector, so
        # nuScenes' 900 queries would make the positive:negative ratio 1:460
        num_query=100,
        LID=True,
        with_position=True,
        with_multiview=True,
        position_range=position_range,
        normedlinear=False,
        # boxes carry no velocity, so 8 = x, y, log dx, log dy, z, log dz,
        # sin yaw, cos yaw
        code_size=8,
        # NOT PETR's uniform [1.0]*8. normalize_bbox leaves cx/cy/cz in raw
        # metres, so at +-160 m the centre terms dominate the L1: the first run
        # with uniform weights held loss_bbox at ~10 (raw L1 ~40 summed over 8
        # dims) and a mean grad_norm of 4706 against clip_grad's max_norm of 35
        # -- a 134x rescale on every step, i.e. the optimiser ran at lr/134.
        # The model underfit into predicting the mean bearing: ground-truth
        # azimuth spread 22.3 deg, predicted spread 3.7 deg, median azimuth
        # error 19 deg. Dividing the centre dims by the half-extent (160 m in
        # x/y, 16 m in z) puts a full-scene-width error at 1.0, the same scale
        # the log-size and sin/cos terms already sit at, so no dim dominates
        # and the clip stops binding.
        code_weights=[
            1 / 160, 1 / 160, 1.0, 1.0, 1 / 16, 1.0, 1.0, 1.0
        ],
        transformer=dict(
            type='PETRTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                        dict(
                            type='PETRMultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.1,
                            dropout_layer=dict(type='Dropout', drop_prob=0.1)),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-180.0, -180.0, -10.0, 180.0, 180.0, 30.0],
            pc_range=point_cloud_range,
            max_num=100,
            num_classes=4),
        positional_encoding=dict(
            type='SinePositionalEncoding3D', num_feats=128, normalize=True),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='mmdet.L1Loss', loss_weight=0.25),
        loss_iou=dict(type='mmdet.GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            grid_size=[1024, 1024, 1],
            voxel_size=[0.3125, 0.3125, 32.0],
            point_cloud_range=point_cloud_range,
            out_size_factor=4,
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                # NOT PETR's BBox3DL1Cost: that costs boxes in raw metres, and
                # at this range the centre terms decide the assignment 185:1
                # over the classification cost, which starves the classifier
                # into a constant score and pins AP at 0. See the docstring on
                # NormalizedBBox3DL1Cost. The weight stays at 0.25 because
                # PETRHead asserts it equals loss_bbox.loss_weight; only the
                # matching is rescaled, the regression loss is untouched.
                reg_cost=dict(
                    type='NormalizedBBox3DL1Cost',
                    weight=0.25,
                    pc_range=point_cloud_range),
                # fake cost, kept only for DETR head compatibility
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))))

train_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=2,
        backend_args=None),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    # only supervise on what the cameras can see -- see the module docstring
    dict(type='ObjectAzimuthFilter', azimuth_range=camera_azimuth_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf, training=True),
    dict(
        type='MaritimeGlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        training=True),
    dict(
        type='Pack3DDetInputs', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadMultiViewImageFromFiles',
        to_float32=True,
        num_views=2,
        backend_args=None),
    dict(
        type='ResizeCropFlipImage', data_aug_conf=ida_aug_conf,
        training=False),
    dict(type='Pack3DDetInputs', keys=['img'])
]

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

# two protocols side by side: the first is comparable with the LiDAR baselines,
# the second is the one that actually measures the method
val_evaluator = [
    dict(type='MaritimeMetric', pcd_limit_range=point_cloud_range,
         prefix='Maritime'),
    dict(
        type='MaritimeMetric',
        pcd_limit_range=point_cloud_range,
        azimuth_range=camera_azimuth_range,
        prefix='MaritimeFOV'),
]
test_evaluator = val_evaluator

lr = 0.0002
epoch_num = 40
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)}),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.33333333,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        begin=0,
        T_max=epoch_num,
        end=epoch_num,
        by_epoch=True,
        eta_min_ratio=1e-3,
        convert_to_iter_based=True),
]

randomness = dict(seed=1, deterministic=False, diff_rank_seed=False)
default_scope = 'mmdet3d'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=3),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
log_level = 'INFO'
load_from = None
resume = False
