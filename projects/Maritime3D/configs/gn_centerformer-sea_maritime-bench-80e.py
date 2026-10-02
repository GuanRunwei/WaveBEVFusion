"""CenterFormer + physical sea-surface supervision, 80 epochs: stage 1 of the
final model on CenterFormer (gn_centerpoint-sea_maritime-bench-80e.py is the
CenterPoint version).

Same data, augmentation, IMU metas, schedule (AdamW 1e-4, 500-iter warmup +
cosine, 8 GPUs x 2 frames), evaluation and LiDAR input (0.1 x 0.1 x 0.4 m
hard voxels, SparseEncoder -> 400 x 400 at 0.8 m) as the CenterPoint
models. CenterFormer (projects/CenterFormer): SECOND / FPN with channel and
spatial attention -> 800 x 800 heatmap at 0.4 m, the 500 best centres as
queries of a deformable transformer over three BEV scales, a 1D head with
IoU rescoring. GroupNorm instead of SyncBN in the 2D layers and the head
(train / test normalisation gap, progress/20260926_181500_BN-train-eval-gap).
The sea surface supervises the queries on GT centres; at test time the boxes
keep the head's heights.
"""
_base_ = ['./gn_centerpoint-sea_maritime-bench-80e.py']

custom_imports = dict(
    imports=['projects.Maritime3D.maritime3d', 'projects.BEVFusion.bevfusion',
             'projects.CenterFormer.centerformer'],
    allow_failed_imports=False)

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
voxel_size = [0.1, 0.1, 0.4]
grid_size = [3200, 3200, 80]
out_size_factor = 4  # heatmap: 800 x 800 at 0.4 m
class_names = ['boat', 'buoy', 'sailboat', 'ship', 'yacht']
tasks = [dict(num_class=5, class_names=class_names)]
gn = dict(type='GN', num_groups=32, eps=1e-3)
sea_surface = dict(
    plane_scale=100.0,
    prior_c=-1.3,
    level_prior_var=1.0,
    wave_sigma=0.3,
    wave_length=25.0,
    tilt_gain=True,
    max_obs=50,
    hist_prob=0.5,
    hist_std_range=(0.05, 1.0),
    kalman_q=0.005,
    heave_gain=0.0,
    max_gap=1.0)

model = dict(
    _delete_=True,
    type='MaritimeCenterFormer',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=10,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(90000, 120000))),
    voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[81, 3200, 3200],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    backbone=dict(
        type='DeformableDecoderRPN',
        layer_nums=[5, 5, 1],
        ds_num_filters=[256, 256, 128],
        num_input_features=512,
        tasks=tasks,
        classes=5,
        use_gt_training=True,
        corner=True,
        assign_label_window_size=1,
        obj_num=500,
        score_threshold=0.01,
        norm_cfg=gn,
        transformer_config=dict(
            depth=2,
            n_heads=6,
            dim_single_head=64,
            dim_ffn=256,
            dropout=0.3,
            out_attn=False,
            n_points=15)),
    bbox_head=dict(
        type='SeaCenterFormerBboxHead',
        in_channels=256,
        tasks=tasks,
        weight=2,
        corner_loss=True,
        iou_loss=True,
        iou_factor=[1, 1, 1, 1, 1],
        assign_label_window_size=1,
        norm_cfg=gn,
        code_weights=[1.0] * 8,
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), iou=(1, 2),
            sig=(1, 2), fb=(1, 2)),
        sea_surface=sea_surface,
        loss_final_height_weight=0.25,
        loss_sigma_weight=0.25,
        loss_level_weight=0.5),
    train_cfg=dict(
        grid_size=grid_size,
        voxel_size=voxel_size,
        out_size_factor=out_size_factor,
        dense_reg=1,
        gaussian_overlap=0.1,
        point_cloud_range=point_cloud_range,
        max_objs=500,
        min_radius=2,
        code_weights=[1.0] * 8),
    test_cfg=dict(
        post_center_limit_range=[-180.0, -180.0, -10.0, 180.0, 180.0, 30.0],
        # per class, as the CenterPoint models (rotated NMS at IoU 0.2,
        # at most 100 boxes per class)
        nms=dict(
            use_rotate_nms=False,
            use_multi_class_nms=True,
            nms_pre_max_size=[1000] * 5,
            nms_post_max_size=[100] * 5,
            nms_iou_threshold=[0.2] * 5),
        score_threshold=0.01,
        pc_range=point_cloud_range[:2],
        out_size_factor=out_size_factor,
        voxel_size=voxel_size[:2],
        obj_num=500))
