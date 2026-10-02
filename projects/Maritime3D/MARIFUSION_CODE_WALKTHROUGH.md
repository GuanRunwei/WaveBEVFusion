# MariFusion 代码解析：LiDAR + 两路前视相机的实例级融合

版本：2026-10-02 · 适合读者：熟悉 PyTorch / mmdet3d，想逐行看懂 MariFusion 的人。

| 文件 | 内容 |
|---|---|
| [maritime3d/mari_fusion.py](maritime3d/mari_fusion.py) | 第二阶段精修头 `RaySplitFusionRefineHead`、共用函数、`MariFusionCenterPoint`，以及对照基线 `BEVFuseCenterPoint` |
| [maritime3d/centerformer_maritime.py](maritime3d/centerformer_maritime.py) | CenterFormer 版本 `MaritimeCenterFormer` |
| [configs/gn_centerpoint-marifusion_maritime-bench.py](configs/gn_centerpoint-marifusion_maritime-bench.py) | 融合消融：普通 CenterPoint 加 MariFusion（20+6 ep） |
| [configs/gn_centerpoint-sea-marifusion_maritime-bench.py](configs/gn_centerpoint-sea-marifusion_maritime-bench.py) | 最终模型：海面监督 CenterPoint 加 MariFusion（80+6 ep） |
| [configs/gn_centerformer-sea-marifusion_maritime-bench.py](configs/gn_centerformer-sea-marifusion_maritime-bench.py) | 最终模型的 CenterFormer 版本 |

文中所有代码块都由 `work_dirs/tmp/docs/build_walkthroughs.py` 直接从源码抽出，行号和当前仓库一致；代码改了以后重跑这个脚本即可同步。海面模型单独写在 [SEA_SURFACE_CODE_WALKTHROUGH.md](SEA_SURFACE_CODE_WALKTHROUGH.md)。

> **术语**：本文的 "MariFusion" 指融合这一部分，在代码里就是第二阶段。完整方法等于"海面监督（训练时） + MariFusion"。

---

## 0. 一句话讲清楚

> **先用 LiDAR 检测器把船找出来；再对每一条找到的船，只去两路前视相机里"它所在的那一小块"取图像信息，用来修它的框。而且只让图像去修它擅长修的那几个量。**

打个比方：LiDAR 像一把卷尺，量距离很准，但远处的船身上只打到几个点，看不清形状和朝向；相机像一双眼睛，看方向和轮廓很准，但单只眼睛判断不了远近。MariFusion 的分工是：**卷尺负责"多远、多高"，眼睛负责"在哪个方向、多大、朝哪边、是不是船"**。

---

## 1. 为什么这样设计：海上场景的四个特点

| 海上场景的特点 | 数据里的事实 | 对应的设计 | 代码位置 |
|---|---|---|---|
| **相机只看前方** | 只有两路前视相机，约 76% 的目标在相机视场外 | 不把整幅图像变成 BEV 去拼接（那样视场外 3/4 的 BEV 格子没有图像），而是**逐个检测**去补图像证据；视场外的目标自动退化成纯 LiDAR | `forward` 中 `gates * (frac > 0)` |
| **水面有倒影** | 船体和灯光会在吃水线正下方形成镜像 | 采样网格的**下边界卡在吃水线**，只多留 2 px，倒影区一律不采 | `_img_feat` 中 `v1 = vwl + wl_margin` |
| **相机测角准、测距差** | 实测（第 5 节）：97 m 处目标横向移动 1 m，图像上移动 20.7 px；沿视线移动 1 m，吃水线只移动 0.69 px | **Ray-split**：沿视线的距离和高度只由 LiDAR 回归；横向、尺寸、朝向、得分由融合特征回归 | `lidar_reg` / `fused_reg` |
| **相机可靠性随场景变化** | 夜间、逆光、雾天、远处小目标 | **逐相机可靠度门控**，加上训练时的模态 dropout | `self.gate`、`drop_all` / `drop_single` |

另外一个工程事实：88% 的视场内目标都落在原图第 256–896 行这条"地平线带"里，所以图像只裁这一条带，裁成 640×2048，保持**全分辨率**（BEVFusion-LC 基线则是先缩放到一半，再裁成 1024×320）。

### 1.1 为什么是实例级融合，而不是 BEVFusion 式的场景级融合

BEVFusion 的做法是"场景级融合"：用 LSS 预测每个像素的深度分布，把整幅图像提升成一张 BEV 特征图，和 LiDAR 的 BEV 图拼接，再送进检测头。它是为 nuScenes 这类自动驾驶场景设计的，背后有几个默认前提；海上场景恰好在这几个前提上都不成立。

**(1) 海上场景和 BEVFusion 的前提不一样**

| | BEVFusion 的前提 | 我们的数据 | 对场景级融合的影响 |
|---|---|---|---|
| **相机覆盖** | 6 路环视，覆盖 360° | 只有 2 路前视，约 76% 的目标在视场外 | BEV 图上约 3/4 的格子没有图像信息，相机通道在那里全是 0，却仍参与整张图的卷积 |
| **深度估计** | 路面、车辆、建筑都有纹理，可以估深度 | 海面几乎没有纹理，目标多在 50–160 m | LSS 要从单目图像估深度。第 5 节实测：97 m 处沿视线方向 1 像素约等于 1.45 m，深度估计误差很大，图像特征会沿视线方向被摊到错误的距离上 |
| **水面倒影** | 基本没有 | 船体和灯光在吃水线下方形成镜像，夜间尤其明显 | LSS 会把倒影像素也提升到 BEV，而且落在错误的位置 |
| **目标占比** | 路上车辆密集 | 大部分画面是水，目标很小、很稀疏 | 场景级融合的大部分计算花在水面上 |
| **图像分辨率** | 下采样后目标仍然够大 | 远处的浮标本来就很小 | 受显存限制只能用低分辨率（BEVFusion-LC 基线是 1024×320），远处的小浮标会缩到十几个像素 |

**(2) 实例级融合怎么对应这些问题**

1. **只在检测到的目标处取图像**：视场外的目标门控严格为 0，就是纯 LiDAR，不会被一堆空的图像通道干扰。
2. **距离不交给图像（ray-split）**：每个目标的距离由 LiDAR 决定，图像只负责它擅长的方位、轮廓和类别。这只有在实例级才做得到：每个实例都有明确的视线方向，才能把修正量拆成"沿视线"和"横跨视线"两部分，分别交给 LiDAR 和融合特征。场景级融合把两种特征先混在一起再送进检测头，没办法规定哪个量由哪个模态决定。
3. **可以屏蔽倒影**：有了 3D 框，就知道吃水线在图像里的哪一行，采样网格的下边界可以正好卡在那里。场景级融合不知道哪些像素是倒影。
4. **可以用全分辨率图像**：只在框的位置采 16 个点，显存开销很小，所以能用 2048×640 的全分辨率地平线带。
5. **可以逐个目标判断相机是否可靠**：比如夜间、逆光、或者目标只有一部分在画面里。
6. **即插即用**：融合作为第二阶段接在任何 LiDAR 检测器后面，可以直接加载 LiDAR 权重；CenterPoint 和 CenterFormer 用的是同一份代码。

**(3) 实验证据（同一起点：CenterPoint 20 ep，都再训 6 ep）**

| | test mAP3D | 视场内 | 视场外 | 参数量 | ms/帧 |
|---|---|---|---|---|---|
| CenterPoint | 34.60 | 25.23 | 38.43 | 8.8 M | 106 |
| + BEV 级融合（BEVFusion 的相机分支，见 4.15 节） | 38.13 | **25.66** | 43.47 | 44.7 M | 162 |
| + 第二阶段，不加相机 | 40.66 | 25.56 | 47.15 | 9.8 M | 109 |
| **+ MariFusion** | **44.04** | **30.33** | 49.79 | 20.2 M | 147 |

最能说明问题的是**视场内**那一列，因为只有视场内的目标才可能从相机得到信息：

- BEV 级融合在视场内几乎没有提升，只多了 +0.4。它在总分上的 +3.5 主要来自视场外，而那里根本没有图像，所以这部分提升更可能来自多训的 6 个 epoch，而不是相机。
- MariFusion 和不加相机的对照相比，视场内提升 +4.8，yacht 提升 +15。
- MariFusion 的参数量不到 BEV 级融合的一半，推理也更快。

**(4) 需要如实说明的地方**

1. **这组对比不完全公平**：BEV 级融合基线用的是 Swin-T 和半分辨率图像（照搬 BEVFusion 原版设置），MariFusion 用的是 ResNet-50 和全分辨率图像，所以一部分差距可能来自分辨率，而不全是融合方式。如果审稿人追问，需要补一个"BEV 级融合也用全分辨率"的对照（LSS 在全分辨率下显存可能放不下）。
2. **实例级融合的代价：召回上限由第一阶段决定。** 第一阶段漏掉的目标，第二阶段救不回来；场景级融合理论上可以发现 LiDAR 检测器漏掉的目标。我们统计了这部分损失有多大（测试集每隔 2 帧取 1 帧，共 3,939 帧，视场内 GT 3,381 个；脚本 `work_dirs/tmp/stage1_recall/`）：

   | 视场内 GT | 20+6 ep 消融模型 | 最终模型（80+6 ep） |
   |---|---|---|
   | **第一阶段没有提议框**（同类、BEV IoU ≥ 0.25）：第二阶段救不回来 | 11.5% | **9.1%** |
   | 　其中 0–50 m / 50–100 m / 100–160 m | 0.0% / 1.9% / 16.3% | 0.0% / 3.5% / 12.1% |
   | 　其中 ship（78 个） | 25.6% | 23.1% |
   | 第一阶段已检测正确（按评估 IoU） | 58.8% | 67.7% |
   | 第二阶段之后检测正确 | 66.3% | 71.6% |
   | 第二阶段修好 / 改坏的目标数 | 339 / 85（净 +254） | 208 / 77（净 +131） |

   怎么读：
   - **约 9–12% 的视场内目标是实例级融合够不到的**，几乎都在 100 m 以外，50 m 以内没有。
   - **这些目标并不是"LiDAR 打不到点"**：漏检目标的点数中位数是 17，所有视场内目标是 19，基本一样；但它们更远（中位数 135 m，整体 115 m）、更长（船长中位数 24 m，整体 16 m）。也就是说，主要是**远处大船只被扫到一部分，第一阶段把中心或尺寸估错了**，而不是完全看不见。对这类目标，LSS 在 135 m 处的深度估计也几乎帮不上忙，所以"场景级融合能找回 LiDAR 漏掉的目标"这个优势，在我们的数据上实际空间不大。
   - 漏检集中在 ship 上（约四分之一），和"远处大船估不准"一致。改进方向在第一阶段（例如对大船的中心和尺寸估计），而不是融合方式。
3. **训充分以后，相机的增益变小了**：在 80 ep 的第一阶段上，MariFusion 只提升 +0.19 的 3D mAP（BEV +2.7），见 7.2 节。上表给出了一部分解释：80 ep 的第一阶段在视场内已经检测正确 67.7%（20 ep 是 58.8%），留给第二阶段修的空间小了，第二阶段的净修正从 +254 个降到 +131 个。

---

## 2. 整体流程

```
                         ┌──────────────── 第一阶段（只用 LiDAR） ────────────────┐
点云 ─► 体素化 ─► SparseEncoder ─► SECOND + FPN ─► BEV 特征 B_L [B,512,400,400] ─► CenterHead ─► 检测框 D¹（不回传梯度）
                                                       │                                            │
                         ┌─────────────── 第二阶段：RaySplitFusionRefineHead ───────────────────────┘
                         │   对每一个检测框 i：
                         │   ① LiDAR 实例特征：在 B_L 上取 中心 + 4 个 BEV 角点（5 个点）
                         │                     + 框自身几何编码            → f_bev,i  [256]
两路图像 ─► ResNet-50 C3/C4 ─► FPN ─► 图像特征 F_I ─┐
  [B,2,3,640,2048]        [2B,256,80,256] [2B,256,40,128]
                         │   ② 反射感知采样：框投影到相机 v → 4×4 网格（下边界 = 吃水线）
                         │      → 以 f_bev,i 为 query 的掩码注意力   → f_img,i,v [256]，有效比例 frac_i,v
                         │   ③ 门控：g_i,v = σ(MLP(frac, 亮度均值, 亮度方差, 距离, 是否可见))；不可见则为 0
                         │      融合：f_i = f_bev,i + Σ_v g_i,v · f_img,i,v
                         │   ④ Ray-split 回归：
                         │        lidar_reg(f_bev,i)  → Δρ（沿视线），Δz
                         │        fused_reg(f_i)      → Δτ（横跨视线），Δlog l,w,h，Δyaw
                         │        quality(f_i)        → q
                         └─► 解码：中心 += 2Δρ·u + 1Δτ·t，尺寸 ×= e^Δ，z += Δz，yaw += Δyaw
                                   分数 = s₁^0.5 · sigmoid(q)^0.5
```

### 2.1 张量形状（CenterPoint 版，每卡 B = 2 帧，2 路相机）

| 名称 | 形状 | 说明 |
|---|---|---|
| `points` | B 个 `[N_i, 4]` | x, y, z, 强度 |
| `imgs` | `[B, 2, 3, 640, 2048]` | 左右前视相机的地平线带，已归一化 |
| `pts_feats[0]` = B_L | `[B, 512, 400, 400]` | 0.8 m/格，覆盖 ±160 m；512 = 两个 FPN 尺度各 256 通道拼接 |
| `img_feats` | `[2B, 256, 80, 256]`、`[2B, 256, 40, 128]` | FPN 两层，stride 分别为 8 和 16 |
| `bright` | `[B, 2, 2]` | 每路相机整条带的亮度均值和方差（在归一化后的像素上算） |
| 第一阶段检测 D¹ | 每帧 n ≤ 500 个框 `[n, 7]` | (x, y, z_底, l, w, h, yaw)，外加分数和类别 |
| 训练提议框 | 每帧 ≤ 128 个检测 + 每个 GT 的一个扰动副本 | 见 `sample_props` |
| `f_bev`、`fused` | `[n, 256]` | 每个实例的特征 |
| `f_img` / `frac` / `gates` | `[2, n, 256]` / `[2, n]` / `[2, n]` | 每路相机分别算 |
| 输出 | `d_range, d_z, d_cross, d_yaw, q`：`[n]`；`d_dim`：`[n, 3]` | 残差和质量分数 |

---

## 3. 配置文件

### 3.1 融合消融配置（普通 CenterPoint 加 MariFusion）

**[gn_centerpoint-marifusion_maritime-bench.py:1-155](configs/gn_centerpoint-marifusion_maritime-bench.py#L1-L155)** · `whole file`

```python
"""MariFusion: CenterPoint (GroupNorm) + two front cameras, two-stage sparse
instance fusion (RaySplitFusionRefineHead): each detection samples the
full-resolution horizon band of both cameras on a reflection-aware grid over
its projection, a per-camera reliability gate (valid fraction, brightness,
range; modality dropout) weights the image evidence, and the residual along
the viewing ray / in height comes from LiDAR only while bearing, size,
heading and score use the fused feature.

Plain CenterHead (no evidence anchor, no sea surface) so the fusion's
contribution is measured alone. Initialised from bench_gn_cp (20 ep) and
trained 6 more epochs; the control without cameras is
gn_centerpoint-refine-lidaronly_maritime-bench.py (same schedule).
"""
_base_ = ['./gn_centerpoint_voxel01_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=True)
# full-resolution horizon band: rows ~256-896 of the 2048 x 1080 images
image_size = [640, 2048]
data_prefix = dict(pts='', CAM_FRONT_LEFT='', CAM_FRONT_RIGHT='')

model = dict(
    type='MariFusionCenterPoint',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        # C3, C4 (strides 8, 16); stage 4 is not built, so every parameter
        # gets a gradient (DDP)
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(1, 2),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024],
        out_channels=256,
        num_outs=2),
    refine_head=dict(
        type='RaySplitFusionRefineHead',
        num_classes=5,
        bev_channels=512,
        img_channels=256,
        hidden=256,
        pc_range=point_cloud_range,
        use_image=True,
        num_levels=2,
        grid=4,
        widen=0.2,
        waterline_margin=2.0,
        drop_all=0.25,
        drop_single=0.1,
        pos_iou=0.25,
        max_train_props=128,
        gt_jitter=True,
        score_alpha=0.5))

meta_keys = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
    'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
    'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats'
]
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=None),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.15, 0.19],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=True),
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
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=meta_keys)
]
test_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        num_views=2,
        backend_args=None),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=None),
    dict(
        type='ImageAug3D',
        final_dim=image_size,
        resize_lim=[1.0, 1.0],
        bot_pct_lim=[0.17, 0.17],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs', keys=['img', 'points'], meta_keys=meta_keys)
]
_ds = dict(
    modality=input_modality,
    data_prefix=data_prefix,
    default_cam_key='CAM_FRONT_LEFT')
train_dataloader = dict(dataset=dict(pipeline=train_pipeline, **_ds))
val_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))
test_dataloader = dict(dataset=dict(pipeline=test_pipeline, **_ds))

# 6 epochs from the LiDAR checkpoint (as BEVFusion-LC); image backbone at
# 0.1x lr
load_from = 'work_dirs/bench_gn_cp/epoch_20.pth'
epoch_num = 6
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=3)
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=dict(img_backbone=dict(lr_mult=0.1))))
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epoch_num, end=epoch_num,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
]
```

逐块说明：

- **`_base_`**：继承普通 CenterPoint（GroupNorm）。LiDAR 部分、CenterHead、数据集、评估方式都不变，所以第二阶段的增益可以干净地归到融合上。
- **`data_preprocessor`**：图像按 ImageNet 均值和方差归一化，BGR 转 RGB，pad 到 32 的倍数（640×2048 本身就整除 32，所以实际不会 pad）。
- **`img_backbone`**：ResNet-50，**只建 3 个 stage**，输出 C3 和 C4。原因是 DDP 要求每个参数在每一步都拿到梯度，如果建了第 4 个 stage 却不用，训练会直接报错。`frozen_stages=1` 冻结 stem 和 layer1；`norm_eval=True` 让 BN 用 ImageNet 预训练的统计量，避免每卡 2 帧时 BN 统计不稳。
- **`img_neck`**：FPN，把 C3 和 C4 都映射到 256 通道。
- **`refine_head`**：第二阶段的全部超参数，逐项含义见 4.3 节的表。
- **管线 `ImageAug3D`**：`resize_lim=[1.0, 1.0]` 表示不缩放；`final_dim=[640, 2048]` 加上 `bot_pct_lim` 就是"裁地平线带"。测试时 `bot_pct=0.17`，裁出第 256–896 行；训练时取 0.15–0.19 之间的随机值，相当于上下抖动约 ±22 px。不做翻转，也不做旋转。
- **点云增强**：`BEVFusionGlobalRotScaleTrans` 和 `BEVFusionRandomFlip3D` 会把增强矩阵记到 `lidar_aug_matrix` 里，投影时要先把这个增强撤销掉（见 `_project`）。
- **训练设置**：`load_from` 加载第一阶段 ep20 的权重，再训 6 个 epoch；图像骨干的学习率乘 0.1；学习率先 warmup，再按 cosine 衰减。

### 3.2 最终模型配置（海面监督 + MariFusion）

和 3.1 的区别：

1. `_base_` 换成海面监督的 CenterPoint，第一阶段的检测头是 `EvidenceSeaCenterHead`，海面损失**在第二阶段训练时继续计算**；
2. meta 里多了 `sea_up / sea_up_valid / sea_up_rot / sea_heave*`，管线里多了 `LoadSeaUp`、`EvidenceAnchor(mode='centre')` 和 `Pack3DDetInputsAnchor`；
3. `load_from` 改为 `bench_gn_cp_sea_80e/epoch_80.pth`。

**[gn_centerpoint-sea-marifusion_maritime-bench.py:1-72](configs/gn_centerpoint-sea-marifusion_maritime-bench.py#L1-L72)** · `lines 1-72`

```python
"""Final model, stage 2: MariFusion on CenterPoint + physical sea surface.

Stage 1 is gn_centerpoint-sea_maritime-bench-80e.py (EvidenceSeaCenterHead,
anchors at the box centre, SeaSurface heights, 80 epochs); this config adds
the two front cameras and the RaySplitFusionRefineHead of
gn_centerpoint-marifusion_maritime-bench.py and trains 6 more epochs from
that checkpoint. The sea-surface losses keep supervising stage 1 during
these epochs; at test time the stage-1 boxes keep the head's own heights
(sea_at_test=False) and the refine head's height residual comes from LiDAR
only.
"""
_base_ = ['./gn_centerpoint-sea_maritime-bench.py']

point_cloud_range = [-160.0, -160.0, -8.0, 160.0, 160.0, 24.0]
input_modality = dict(use_lidar=True, use_camera=True)
image_size = [640, 2048]
data_prefix = dict(pts='', CAM_FRONT_LEFT='', CAM_FRONT_RIGHT='')
sea_up_lookup = 'dataset/infos/detection/sea_up_lidar.npz'
sea_heave_lookup = 'dataset/infos/detection/sea_heave_lidar.npz'

model = dict(
    type='MariFusionCenterPoint',
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=3,
        strides=(1, 2, 2),
        dilations=(1, 1, 1),
        out_indices=(1, 2),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet50')),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[512, 1024],
        out_channels=256,
        num_outs=2),
    refine_head=dict(
        type='RaySplitFusionRefineHead',
        num_classes=5,
        bev_channels=512,
        img_channels=256,
        hidden=256,
        pc_range=point_cloud_range,
        use_image=True,
        num_levels=2,
        grid=4,
        widen=0.2,
        waterline_margin=2.0,
        drop_all=0.25,
        drop_single=0.1,
        pos_iou=0.25,
        max_train_props=128,
        gt_jitter=True,
        score_alpha=0.5))

# MariFusion's keys + LoadSeaUp's
meta_keys = [
    'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
    'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
    'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
    'pcd_scale_factor', 'pcd_trans', 'lidar_aug_matrix', 'num_pts_feats',
    'sea_up', 'sea_up_valid', 'sea_heave', 'sea_heave_valid', 'sea_up_rot'
]
```

CenterFormer 版本（`gn_centerformer-sea-marifusion_maritime-bench.py`）的结构完全一样，只有两处不同：`bev_channels=256`（CenterFormer 预测 heatmap 用的那张 BEV 图是 256 通道、800×800），`load_from` 指向 CenterFormer 第一阶段的权重。

---

## 4. 代码逐段解析（`mari_fusion.py`）

### 4.1 模块说明与依赖

**[mari_fusion.py:1-39](maritime3d/mari_fusion.py#L1-L39)** · `lines 1-39`

```python
"""MariFusion: two-stage sparse instance fusion of LiDAR and the two front
cameras on top of CenterPoint.

Stage 1 is CenterPoint (LiDAR only). Stage 2 refines every detection with
image evidence, sampled only where the detection projects:

* image sampling -- each detection's 3D box is projected into both front
  cameras (full-resolution horizon band); a G x G grid over its projection,
  widened sideways and upwards for box errors but never extended below the
  projected waterline (the bottom face), samples two FPN levels. Water
  mirrors hulls and lights directly below the waterline, so that region is
  masked out (reflection-aware sampling);
* per-camera reliability gate -- g_v = sigmoid(MLP(valid fraction, image
  brightness, range)), zero when nothing projects into camera v; modality
  dropout in training (all cameras p_all, one camera p_single). Outside the
  cameras (76% of the objects) the refinement is LiDAR-only by construction;
* ray-split refinement -- the residual along the viewing ray (range) and in
  height come from the LiDAR BEV feature alone; the residual across the ray
  (bearing), the size, the heading and the score come from the fused
  feature. A camera pins angles (1 px ~ 5.5 cm across at 100 m) but not
  range (1 px of waterline ~ 1.8 m along it); LiDAR is the opposite.

``use_image=False`` gives the same second stage without cameras (the
control that separates the camera's contribution from the refinement's).
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.detectors import CenterPoint
from mmdet3d.registry import MODELS
```

模块 docstring 就是设计摘要，和第 1 节对应。`box_iou_rotated` 是 mmcv 的 BEV 旋转框 IoU，用来给提议框配 GT。

### 4.2 两个小工具：`_mlp` 和 `_corners`

**[mari_fusion.py:42-48](maritime3d/mari_fusion.py#L42-L48)** · `_mlp`

```python
def _mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)
```

`_mlp(i, h, o)` 生成一个两层 MLP：`Linear(i→h) → LayerNorm → ReLU → Linear(h→o)`。这里**用 LayerNorm 而不是 BatchNorm**：第二阶段每帧的实例数从 0 到几百不等，BN 的统计量会极不稳定，而且训练和推理之间会有差异（这正是之前 BEV 主干出过的问题）。

**[mari_fusion.py:51-62](maritime3d/mari_fusion.py#L51-L62)** · `_corners`

```python
def _corners(b: Tensor) -> Tensor:
    """[n, 7] boxes (x, y, z_bottom, l, w, h, yaw) -> [n, 8, 3] corners;
    0-3 bottom, 4-7 top."""
    x, y, z, l, w, h, yaw = b.unbind(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    du = torch.stack([l, l, -l, -l], -1) / 2
    dv = torch.stack([w, -w, -w, w], -1) / 2
    cx = x[:, None] + du * c[:, None] - dv * s[:, None]
    cy = y[:, None] + du * s[:, None] + dv * c[:, None]
    bot = torch.stack([cx, cy, z[:, None].expand_as(cx)], -1)
    top = torch.stack([cx, cy, (z + h)[:, None].expand_as(cx)], -1)
    return torch.cat([bot, top], 1)
```

`_corners` 把 7 维框 `(x, y, z_底, l, w, h, yaw)` 变成 8 个角点。注意 **mmdet3d 的 LiDAR 框 z 是底面高度**，所以 0–3 号是底面四角（吃水线所在的那一圈），4–7 号是顶面。BEV 俯视下的角点顺序是：

```
          yaw 方向（船头）→
   3 ───────────── 0          du = ±l/2（沿船长方向）
   │               │          dv = ±w/2（沿船宽方向）
   2 ───────────── 1
```

### 4.3 `RaySplitFusionRefineHead.__init__`：参数和子模块

**[mari_fusion.py:91-140](maritime3d/mari_fusion.py#L91-L140)** · `RaySplitFusionRefineHead.__init__`

```python
    def __init__(self,
                 num_classes: int = 5,
                 bev_channels: int = 512,
                 img_channels: int = 256,
                 hidden: int = 256,
                 pc_range=(-160.0, -160.0, -8.0, 160.0, 160.0, 24.0),
                 use_image: bool = True,
                 num_levels: int = 2,
                 grid: int = 4,
                 widen: float = 0.2,
                 waterline_margin: float = 2.0,
                 drop_all: float = 0.25,
                 drop_single: float = 0.1,
                 pos_iou: float = 0.25,
                 max_train_props: int = 128,
                 gt_jitter: bool = True,
                 score_alpha: float = 0.5,
                 range_scale: float = 2.0,
                 cross_scale: float = 1.0,
                 loss_reg_weight: float = 1.0,
                 loss_q_weight: float = 1.0,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.pc_range = [float(v) for v in pc_range]
        self.use_image = use_image
        self.num_levels = num_levels
        self.grid = grid
        self.widen = widen
        self.wl_margin = waterline_margin
        self.drop_all, self.drop_single = drop_all, drop_single
        self.pos_iou = pos_iou
        self.max_train_props = max_train_props
        self.gt_jitter = gt_jitter
        self.alpha = score_alpha
        self.range_scale, self.cross_scale = range_scale, cross_scale
        self.w_reg, self.w_q = loss_reg_weight, loss_q_weight
        # LiDAR instance feature: centre + 4 BEV corners of the box
        self.bev_proj = _mlp(5 * bev_channels, hidden, hidden)
        self.box_enc = _mlp(10 + num_classes, hidden, hidden)
        if use_image:
            self.img_proj = nn.Linear(num_levels * img_channels, hidden)
            self.img_pos = _mlp(2, hidden // 2, hidden)
            self.q_proj = nn.Linear(hidden, hidden)
            self.img_out = nn.Linear(hidden, hidden)
            self.gate = _mlp(5, 64, 1)
        self.lidar_reg = _mlp(hidden, hidden, 2)       # d_range, d_z
        self.fused_reg = _mlp(hidden, hidden, 5)       # d_cross, dlog lwh, d_yaw
        self.quality = _mlp(hidden, hidden, 1)
        self._last_gate = None
```

**超参数**（括号里是配置中的取值）：

| 参数 | 含义 |
|---|---|
| `bev_channels` (512；CenterFormer 为 256) | 第一阶段 BEV 特征的通道数 |
| `num_levels` (2)、`img_channels` (256) | 采样的 FPN 层数和每层通道数 |
| `grid` (4) | 每个框、每路相机采 4×4 = 16 个点 |
| `widen` (0.2) | 网格向左、向右、向上各外扩投影框尺寸的 20%，用来容忍第一阶段框的误差 |
| `waterline_margin` (2.0) | 吃水线以下只多留 2 像素 |
| `drop_all` (0.25)、`drop_single` (0.1) | 训练时以 25% 的概率关掉全部相机，并对每个实例以 10% 的概率关掉某一路相机 |
| `pos_iou` (0.25) | 提议框和 GT 的 BEV IoU ≥ 0.25 才回归残差 |
| `max_train_props` (128) | 训练时每帧最多取 128 个第一阶段检测 |
| `gt_jitter` (True) | 训练时为每个 GT 额外加一个扰动副本作为提议 |
| `score_alpha` (0.5) | 最终分数 = s₁^0.5 · sigmoid(q)^0.5 |
| `range_scale` (2.0)、`cross_scale` (1.0) | 残差每个单位对应的米数：沿视线 2 m，横向 1 m |

**子模块**（参数量为 CenterPoint 版的实测值，合计 1.29 M）：

| 子模块 | 输入 → 输出 | 参数量 | 作用 |
|---|---|---|---|
| `bev_proj` | 5×512 → 256 | 721,920 | 把 5 个 BEV 采样点的特征压成实例特征 |
| `box_enc` | 15 → 256 | 70,400 | 编码框自身的几何（10 维）+ 类别（5 维 one-hot） |
| `img_proj` | 2×256 → 256 | 131,328 | 把两层 FPN 的采样特征拼起来再投影 |
| `img_pos` | 2 → 256 | 33,664 | 采样点在网格内的相对位置编码 |
| `q_proj`、`img_out` | 256 → 256 | 各 65,792 | 注意力的 query 投影和输出投影 |
| `gate` | 5 → 1 | 577 | 每路相机的可靠度门控 |
| `lidar_reg` | 256 → 2 | 66,818 | **只吃 LiDAR 特征**，输出 Δρ、Δz |
| `fused_reg` | 256 → 5 | 67,589 | 吃融合特征，输出 Δτ、Δlog(l, w, h)、Δyaw |
| `quality` | 256 → 1 | 66,561 | 吃融合特征，输出质量分数 q |

`use_image=False` 时不会建图像相关的 5 个子模块，这就是"二阶段不加相机"的对照模型（`gn_centerpoint-refine-lidaronly`）。`self._last_gate` 目前没有被使用，属于残留。

### 4.4 视线坐标系：`_ray`

**[mari_fusion.py:143-147](maritime3d/mari_fusion.py#L143-L147)** · `RaySplitFusionRefineHead._ray`

```python
    def _ray(self, b):
        r = torch.hypot(b[:, 0], b[:, 1]).clamp_min(1e-3)
        u = torch.stack([b[:, 0], b[:, 1]], -1) / r[:, None]
        t = torch.stack([-u[:, 1], u[:, 0]], -1)
        return r, u, t
```

对每个框中心 (x, y) 计算：
- `r`：到本船（LiDAR 原点）的距离；
- `u = (x, y)/r`：**沿视线**的单位向量，从本船指向目标；
- `t = (−u_y, u_x)`：**横跨视线**的单位向量，即 u 逆时针转 90°。

后面所有 ray-split 的分解（目标、扰动、解码）都在 (u, t) 这个以目标为中心的局部坐标系里做。

### 4.5 LiDAR 实例特征：`_bev_feat` 和 `_box_enc`

**[mari_fusion.py:149-156](maritime3d/mari_fusion.py#L149-L156)** · `RaySplitFusionRefineHead._bev_feat`

```python
    def _bev_feat(self, bev: Tensor, b: Tensor) -> Tensor:
        """[C, H, W] map, [n, 7] boxes -> [n, 5 C] (centre + BEV corners)."""
        pts = torch.cat([b[:, None, :2], _corners(b)[:, :4, :2]], 1)
        x0, y0, _, x1, y1, _ = self.pc_range
        g = torch.stack([(pts[..., 0] - x0) / (x1 - x0) * 2 - 1,
                         (pts[..., 1] - y0) / (y1 - y0) * 2 - 1], -1)
        f = F.grid_sample(bev[None].float(), g[None], align_corners=False)
        return f[0].permute(1, 2, 0).reshape(b.shape[0], -1)
```

- 取 5 个点：框中心加底面 4 个角点的 (x, y)。只取中心会丢掉"框的轮廓到底有没有压在点云上"的信息；把 4 个角也取上，长船的首尾都能读到特征。
- 把米制坐标线性映射到 `[−1, 1]`，用 `grid_sample` 做双线性插值。`align_corners=False` 时，`−1` 对应最左格子的左边缘，和 BEV 网格"第 i 格中心在 x₀ + (i+0.5)·0.8 m"正好一致。
- 特征图的列对应 x、行对应 y，和 CenterPoint 的 heatmap 索引 `ind = y·W + x` 一致。
- 输出 `[n, 5·C]`。

**[mari_fusion.py:158-168](maritime3d/mari_fusion.py#L158-L168)** · `RaySplitFusionRefineHead._box_enc`

```python
    def _box_enc(self, b, s, lab):
        r, u, _ = self._ray(b)
        theta = torch.atan2(u[:, 1], u[:, 0])
        rel = b[:, 6] - theta
        e = torch.stack([
            r / 160.0, torch.sin(theta), torch.cos(theta), b[:, 2] / 8.0,
            b[:, 3].clamp_min(1e-2).log(), b[:, 4].clamp_min(1e-2).log(),
            b[:, 5].clamp_min(1e-2).log(), torch.sin(rel), torch.cos(rel), s
        ], -1)
        return torch.cat([e, F.one_hot(lab.long(), self.num_classes).float()],
                         -1)
```

10 维几何编码，再拼 5 维类别 one-hot：

| 维 | 内容 | 为什么要 |
|---|---|---|
| 1 | r / 160 | 距离：远处的点更稀、像素更少，修正策略应该不同 |
| 2–3 | sin θ, cos θ | 目标在本船的哪个方位 |
| 4 | z / 8 | 底面高度 |
| 5–7 | log l, log w, log h | 尺寸，取 log 使大船和小船处在同一量级 |
| 8–9 | sin(yaw − θ), cos(yaw − θ) | **相对视线的朝向**：侧对和正对 LiDAR 时，可见轮廓完全不同 |
| 10 | s₁ | 第一阶段分数 |

`f_bev = bev_proj(5 点特征) + box_enc(几何编码)`，两者相加得到实例的 LiDAR 表示，后面既作为纯 LiDAR 回归的输入，也作为看图像时的 query。

### 4.6 3D 点投影到图像：`_project`

**[mari_fusion.py:170-184](maritime3d/mari_fusion.py#L170-L184)** · `RaySplitFusionRefineHead._project`

```python
    def _project(self, pts, meta, v):
        """[m, 3] augmented-LiDAR points -> pixel (u, v) [m, 2], depth [m]."""
        dev = pts.device
        l2i = torch.as_tensor(meta['lidar2img'][v], dtype=torch.float32,
                              device=dev)
        aug = torch.as_tensor(meta.get('lidar_aug_matrix', torch.eye(4)),
                              dtype=torch.float32, device=dev)
        ia = torch.as_tensor(meta['img_aug_matrix'][v], dtype=torch.float32,
                             device=dev)
        hom = torch.cat([pts, torch.ones_like(pts[:, :1])], -1)
        q = (l2i @ torch.linalg.inv(aug) @ hom.T).T
        d = q[:, 2]
        uv = q[:, :2] / d.clamp_min(1e-3)[:, None]
        uv = uv @ ia[:2, :2].T + ia[:2, 3]
        return uv, d
```

坐标链：

```
增强后的 LiDAR 坐标 ──inv(lidar_aug_matrix)──► 原始 LiDAR 坐标 ──lidar2img──► 原图齐次坐标 (u·d, v·d, d)
   ──除以深度 d──► 原图像素 ──img_aug_matrix（裁剪 / 缩放）──► 网络输入图像上的像素 (u, v)
```

- 第一阶段的框是在**增强后**的点云坐标系里预测的，所以先用 `inv(lidar_aug_matrix)` 撤销旋转、缩放和翻转，才能用原始标定。
- `lidar2img` 是 4×4 矩阵，等于内参乘外参，来自标定文件。
- 深度 `d` 先 `clamp_min(1e-3)` 再做除法，防止除零；相机背后的点（d ≤ 0.5）在 `_img_feat` 里会被标为无效。
- 这个公式和 BEVFusion-LC 基线用的是同一套投影链路，测试脚本 `work_dirs/tmp/test_marifusion.py` 核对过两者，最大误差 0.15 px。

### 4.7 反射感知采样 + 掩码注意力：`_img_feat`（核心）

**[mari_fusion.py:186-240](maritime3d/mari_fusion.py#L186-L240)** · `RaySplitFusionRefineHead._img_feat`

```python
    def _img_feat(self, feats, meta, b, q, bright, img_hw):
        """Masked attention over the reflection-aware grid, per camera.

        feats: list over levels of [N_cam, C, h, w]. Returns image features
        [N_cam, n, hidden] and the valid fraction [N_cam, n].
        """
        n, G = b.shape[0], self.grid
        H, W = img_hw
        out, frac = [], []
        cor = _corners(b)                                   # [n, 8, 3]
        for v in range(feats[0].shape[0]):
            uv, d = self._project(cor.reshape(-1, 3), meta, v)
            uv, d = uv.view(n, 8, 2), d.view(n, 8)
            ok_c = d > 0.5
            big = torch.full_like(uv[..., 0], 1e6)
            umin = torch.where(ok_c, uv[..., 0], big).min(1).values
            umax = torch.where(ok_c, uv[..., 0], -big).max(1).values
            vmin = torch.where(ok_c, uv[..., 1], big).min(1).values
            # waterline: the lowest projected bottom corner
            vwl = torch.where(ok_c[:, :4], uv[:, :4, 1], -big[:, :4]).max(1) \
                .values
            du = (umax - umin) * self.widen
            u0, u1 = umin - du, umax + du
            v0 = vmin - (vwl - vmin) * self.widen
            v1 = vwl + self.wl_margin
            lin = (torch.arange(G, device=b.device).float() + 0.5) / G
            gu = u0[:, None] + (u1 - u0)[:, None] * lin[None]    # [n, G]
            gv = v0[:, None] + (v1 - v0)[:, None] * lin[None]
            gu = gu[:, None, :].expand(n, G, G).reshape(n, G * G)
            gv = gv[:, :, None].expand(n, G, G).reshape(n, G * G)
            valid = (ok_c.sum(1) >= 4)[:, None] & (gu >= 0) & (gu < W) & \
                (gv >= 0) & (gv < H)
            # masked samples must not carry NaN / inf into the attention
            gu = torch.nan_to_num(gu).clamp(-W, 2 * W)
            gv = torch.nan_to_num(gv).clamp(-H, 2 * H)
            u0 = torch.nan_to_num(u0).clamp(-W, 2 * W)
            u1 = torch.nan_to_num(u1).clamp(-W, 2 * W)
            v0 = torch.nan_to_num(v0).clamp(-H, 2 * H)
            v1 = torch.nan_to_num(v1).clamp(-H, 2 * H)
            grid = torch.stack([gu / W * 2 - 1, gv / H * 2 - 1], -1)
            smp = [F.grid_sample(f[v:v + 1].float(), grid[None],
                                 align_corners=False)[0].permute(1, 2, 0)
                   for f in feats[:self.num_levels]]           # [n, G2, C]
            k = self.img_proj(torch.cat(smp, -1))
            pos = torch.stack([(gu - u0[:, None]) /
                               (u1 - u0).clamp_min(1)[:, None],
                               (gv - v0[:, None]) /
                               (v1 - v0).clamp_min(1)[:, None]], -1)
            k = (k + self.img_pos(pos.clamp(0, 1))) * valid[..., None]
            att = (self.q_proj(q)[:, None, :] * k).sum(-1) / math.sqrt(
                k.shape[-1])
            att = att.masked_fill(~valid, -1e4).softmax(-1) * valid
            out.append(self.img_out((att[..., None] * k).sum(1)))
            frac.append(valid.float().mean(1))
        return torch.stack(out), torch.stack(frac)
```

逐步说明（对每路相机 v 分别做）：

**① 投影 8 个角点**（第 197–199 行）。深度大于 0.5 m 的角点记为可用（`ok_c`）。

**② 求投影框和吃水线**（第 200–206 行）。只在可用角点上求 `umin / umax / vmin`；`big = 1e6` 用来把不可用的角点排除在 min/max 之外。**吃水线行 `vwl` = 底面 4 个角点投影后最靠下的那一行**（图像 v 轴向下为正，所以取 max）。

**③ 确定采样区域**（第 207–210 行）：

```
                 v0 = vmin − 0.2·(vwl − vmin)     ← 向上外扩 20%（桅杆、上层建筑可能超出框）
        u0 ┌──────────────────────────────┐ u1   ← 左右各外扩投影宽度的 20%
           │   ·      ·      ·      ·     │
           │   ·    ▄▄███▄▄  ·      ·     │       · = 4×4 采样点（取格子中心）
           │   ·   ████████▄▄▄▄  ·  ·     │
           │   ·      ·      ·      ·     │
   ════════└══════════════════════════════┘═════ v1 = vwl + 2 px   ← 吃水线，下边界
           ░░░░░░░░ 倒影区：不采样 ░░░░░░░░
```

上、左、右都放宽，用来容忍第一阶段框的误差；**只有下边界不放宽**，因为吃水线以下就是倒影，采到的是"镜子里的船"，会把高度和尺寸带偏。

**④ 生成 4×4 网格**（第 211–215 行）。`lin = (k + 0.5)/G` 取每个子格的中心，`gu` 和 `gv` 展开成 16 个点。

**⑤ 有效性掩码**（第 216–217 行）。一个采样点有效，必须同时满足：至少 4 个角点在相机前方；这个点落在图像范围内。

**⑥ 数值清理**（第 219–224 行）。无效点的坐标可能是 inf 或 NaN（例如目标在相机背后）。虽然后面会被掩码掉，但 `grid_sample` 和 0 × inf 依然会产生 NaN 并污染梯度，所以先清理、再裁剪到合理范围。这是调试时真实遇到过的 bug。

**⑦ 采样两层 FPN 并编码**（第 225–234 行）：
- 归一化到 `[−1, 1]` 后，在 stride 8 和 stride 16 两层上用 `grid_sample` 采样，拼接成 512 维，经 `img_proj` 映射到 256 维；
- 再加上 `img_pos(pos)`，`pos` 是采样点在网格内的相对位置 (0–1, 0–1)，让网络知道"这是船头那一侧"还是"靠近吃水线"；
- 最后乘以 `valid`，无效点的特征置零。

**⑧ 掩码注意力**（第 235–238 行）。以 `f_bev`（经 `q_proj` 投影）为 query，16 个采样点为 key 和 value，做缩放点积注意力；无效点先填 −1e4 再做 softmax，之后再乘一次 `valid`。所以**一个点都不可见时，输出严格为 0**，而不是 16 个无效点的平均。输出经 `img_out` 投影，得到 `f_img,i,v`。

**⑨ 有效比例** `frac = valid.mean()`：投影进相机的采样点占多少（0–1），作为门控的一个输入。

### 4.8 前向：门控、模态 dropout、融合、ray-split 回归

**[mari_fusion.py:243-285](maritime3d/mari_fusion.py#L243-L285)** · `RaySplitFusionRefineHead.forward`

```python
    def forward(self, props: List[Tensor], scores: List[Tensor],
                labels: List[Tensor], bev: Tensor,
                img_feats: Optional[List[Tensor]], metas: List[dict],
                bright: Optional[Tensor], img_hw=None) -> List[dict]:
        """Per frame: residuals, quality logit and gates of its proposals."""
        outs = []
        for i, (b, s, lab) in enumerate(zip(props, scores, labels)):
            if b.shape[0] == 0:
                outs.append(None)
                continue
            b = self.sanitize(b.float())
            f_bev = self.bev_proj(self._bev_feat(bev[i], b)) + \
                self.box_enc(self._box_enc(b, s.float(), lab))
            fused = f_bev
            gates = None
            if self.use_image and img_feats is not None:
                n_cam = len(metas[i]['lidar2img'])
                lv = [f[i * n_cam:(i + 1) * n_cam] for f in img_feats]
                f_img, frac = self._img_feat(lv, metas[i], b, f_bev,
                                             bright[i], img_hw)
                r = torch.hypot(b[:, 0], b[:, 1]) / 160.0
                gin = torch.stack([
                    frac, bright[i][:, None, 0].expand_as(frac),
                    bright[i][:, None, 1].expand_as(frac),
                    r[None].expand_as(frac), (frac > 0).float()
                ], -1)
                gates = torch.sigmoid(self.gate(gin))[..., 0] * (frac > 0)
                if self.training:
                    keep = torch.ones_like(gates[:, :1])
                    if torch.rand(()) < self.drop_all:
                        keep = keep * 0
                    single = (torch.rand(gates.shape[0], 1,
                                         device=gates.device) >=
                              self.drop_single).float()
                    gates = gates * keep * single
                fused = f_bev + (gates[..., None] * f_img).sum(0)
            lr = self.lidar_reg(f_bev)
            fr = self.fused_reg(fused)
            outs.append(dict(d_range=lr[:, 0], d_z=lr[:, 1],
                             d_cross=fr[:, 0], d_dim=fr[:, 1:4],
                             d_yaw=fr[:, 4], q=self.quality(fused)[:, 0],
                             gates=gates))
        return outs
```

按帧循环（每帧的实例数不同，不 pad，代码更直观；实例数最多几百，开销很小）：

1. **空帧**直接输出 `None`（第 250–252 行）。
2. **清理框**（`sanitize`，见 4.9），然后算 `f_bev`（第 253–255 行）。
3. **取这一帧的图像特征**（第 259–260 行）：图像分支把 `[B, 2, ...]` 展平成 `[2B, ...]`，所以第 i 帧对应第 `i·2` 到 `i·2+1` 行。
4. **门控输入**（第 263–268 行），每路相机 5 个数：有效比例 `frac`、整条带亮度的均值和方差（夜间整体暗、逆光时方差大）、距离 r/160、是否可见 `frac > 0`。
5. **门控**（第 269 行）：`g = sigmoid(MLP(...)) · (frac > 0)`。**不可见的相机门控严格为 0**，所以约 76% 的视场外目标，其融合特征精确等于 `f_bev`。
6. **模态 dropout**（第 270–277 行，只在训练时）：25% 的概率整帧所有门控置 0；另外每个实例以 10% 的概率把某一路相机置 0。这样网络不能依赖相机，夜间或相机失效时也能稳定退化为 LiDAR。
7. **融合**（第 278 行）：`fused = f_bev + Σ_v g_v · f_img,v`，是**残差式**的加法：图像只是在 LiDAR 特征上"补"一点信息，门控为 0 时就是纯 LiDAR。
8. **Ray-split 回归**（第 279–284 行）：
   - `lidar_reg(f_bev)` → `d_range`（沿视线）、`d_z`（高度）——**这两个量的计算路径里根本没有图像特征**；
   - `fused_reg(fused)` → `d_cross`（横跨视线）、`d_dim`（3 个 log 尺寸）、`d_yaw`；
   - `quality(fused)` → `q`（质量 logit）。

### 4.9 解码：`sanitize`、`refine_boxes`、`score`

**[mari_fusion.py:287-293](maritime3d/mari_fusion.py#L287-L293)** · `RaySplitFusionRefineHead.sanitize`

```python
    @staticmethod
    def sanitize(b: Tensor) -> Tensor:
        """Finite boxes with sizes in [0.1, 200] m (stage-1 decodes of an
        untrained model can overflow)."""
        b = torch.nan_to_num(b, nan=0.0, posinf=1e3, neginf=-1e3)
        return torch.cat([b[:, :3].clamp(-1e3, 1e3),
                          b[:, 3:6].clamp(0.1, 200.0), b[:, 6:7]], -1)
```

把第一阶段框里的 NaN 或 inf 替换掉，坐标裁到 ±1000 m，尺寸裁到 0.1–200 m。第二阶段刚接上、第一阶段还没训好时，解码出来的框可能溢出，这一步是防御性的。

**[mari_fusion.py:295-303](maritime3d/mari_fusion.py#L295-L303)** · `RaySplitFusionRefineHead.refine_boxes`

```python
    def refine_boxes(self, b: Tensor, o: dict) -> Tensor:
        """Refined [n, 7] boxes (bottom z)."""
        b = self.sanitize(b)
        _, u, t = self._ray(b)
        c = b[:, :2] + (o['d_range'] * self.range_scale)[:, None] * u + \
            (o['d_cross'] * self.cross_scale)[:, None] * t
        dims = b[:, 3:6] * o['d_dim'].clamp(-1, 1).exp()
        return torch.cat([c, (b[:, 2] + o['d_z'])[:, None], dims,
                          (b[:, 6] + o['d_yaw'])[:, None]], -1)
```

```
中心  c' = c + (2.0 · Δρ) · u + (1.0 · Δτ) · t        ← 沿视线 2 m/单位，横向 1 m/单位
尺寸  (l,w,h)' = (l,w,h) · exp(clamp(Δdim, −1, 1))     ← 单步最多放大 e≈2.7 倍或缩小到 1/2.7
高度  z' = z + Δz
朝向  yaw' = yaw + Δyaw
```

沿视线的缩放取 2 m，是因为 LiDAR 在距离方向上的典型误差比横向大。

**[mari_fusion.py:305-307](maritime3d/mari_fusion.py#L305-L307)** · `RaySplitFusionRefineHead.score`

```python
    def score(self, s1: Tensor, o: dict) -> Tensor:
        return s1.clamp_min(1e-6)**(1 - self.alpha) * \
            torch.sigmoid(o['q'])**self.alpha
```

**最终分数 = 第一阶段分数^0.5 × sigmoid(q)^0.5**，即几何平均。第二阶段只能"调"分数，不能完全推翻第一阶段：第一阶段分数很低的框，即使 q 很高也上不去，反之亦然。

### 4.10 训练用的提议框：`sample_props`

**[mari_fusion.py:310-337](maritime3d/mari_fusion.py#L310-L337)** · `RaySplitFusionRefineHead.sample_props`

```python
    def sample_props(self, dets: List[InstanceData],
                     gts: List[InstanceData]):
        """Stage-1 detections (top-K) plus one jittered copy of each GT."""
        props, scores, labels = [], [], []
        for d, g in zip(dets, gts):
            b = d.bboxes_3d.tensor[:, :7].detach()
            s = d.scores_3d.detach()
            lab = d.labels_3d.detach()
            if b.shape[0] > self.max_train_props:
                k = s.topk(self.max_train_props).indices
                b, s, lab = b[k], s[k], lab[k]
            gb = g.bboxes_3d.tensor[:, :7].to(b.device).float()
            if self.gt_jitter and gb.shape[0]:
                _, u, t = self._ray(gb)
                sig = (gb[:, 3:5].max(1).values / 10).clamp(0.3, 3.0)
                j = gb.clone()
                j[:, :2] += (torch.randn_like(sig) * sig)[:, None] * u + \
                    (torch.randn_like(sig) * sig * 0.5)[:, None] * t
                j[:, 2] += torch.randn_like(sig) * 0.3
                j[:, 3:6] *= (torch.randn_like(j[:, 3:6]) * 0.1).exp()
                j[:, 6] += torch.randn_like(sig) * 0.1
                b = torch.cat([b, j])
                s = torch.cat([s, torch.rand_like(sig) * 0.8 + 0.1])
                lab = torch.cat([lab, g.labels_3d.to(lab.device).long()])
            props.append(b)
            scores.append(s)
            labels.append(lab)
        return props, scores, labels
```

每帧的提议框 = **第一阶段检测按分数取前 128** + **每个 GT 的一个扰动副本**。扰动副本保证即使第一阶段漏检，第二阶段也有"框偏了该怎么修"的正样本；扰动的形状按 ray-split 的思路设计：

| 量 | 扰动 | 说明 |
|---|---|---|
| 沿视线 | N(0, σ²)，σ = max(l, w)/10，并裁到 0.3–3 m | 大船偏得多，小船偏得少 |
| 横向 | N(0, (σ/2)²) | 横向误差通常更小 |
| 高度 | N(0, 0.3²) m | |
| 尺寸 | × exp(N(0, 0.1²)) | 约 ±10% |
| 朝向 | N(0, 0.1²) rad | 约 ±6° |
| 分数 | U(0.1, 0.9) | 让质量头见到各种第一阶段分数 |

所有提议框都 `detach`：第二阶段不会通过框坐标把梯度传回第一阶段的头。

### 4.11 损失：`loss`

**[mari_fusion.py:339-384](maritime3d/mari_fusion.py#L339-L384)** · `RaySplitFusionRefineHead.loss`

```python
    def loss(self, outs, props, labels, gts) -> dict:
        reg, qls, n_pos = [], [], 0
        for o, b, lab, g in zip(outs, props, labels, gts):
            if o is None:
                continue
            b = self.sanitize(b.float())
            gb = g.bboxes_3d.tensor[:, :7].to(b.device).float()
            gl = g.labels_3d.to(b.device)
            if gb.shape[0] == 0:
                qls.append(F.binary_cross_entropy_with_logits(
                    o['q'], torch.zeros_like(o['q']), reduction='sum'))
                continue
            iou = box_iou_rotated(self._bev(b), self._bev(gb))   # [n, m]
            best, j = iou.max(1)
            same = (gl[None, :] == lab[:, None]).float()
            q_t = ((iou * same).max(1).values - 0.25).div(0.5).clamp(0, 1)
            qls.append(F.binary_cross_entropy_with_logits(
                o['q'], q_t, reduction='sum'))
            pos = best >= self.pos_iou
            if not pos.any():
                continue
            bp, gp = b[pos], gb[j[pos]]
            _, u, t = self._ray(bp)
            dc = gp[:, :2] - bp[:, :2]
            dyaw = (gp[:, 6] - bp[:, 6] + math.pi / 2) % math.pi - math.pi / 2
            tgt = torch.stack([
                (dc * u).sum(1) / self.range_scale, gp[:, 2] - bp[:, 2],
                (dc * t).sum(1) / self.cross_scale,
                *(gp[:, 3:6] / bp[:, 3:6]).clamp_min(1e-3).log().unbind(1),
                dyaw
            ], 1)
            pred = torch.stack([
                o['d_range'][pos], o['d_z'][pos], o['d_cross'][pos],
                *o['d_dim'][pos].unbind(1), o['d_yaw'][pos]
            ], 1)
            reg.append(F.smooth_l1_loss(pred, tgt, beta=0.1,
                                        reduction='none').sum(1))
            n_pos += int(pos.sum())
        n_all = sum(o['q'].numel() for o in outs if o is not None)
        dev = next(self.parameters()).device
        zero = sum(p.sum() for p in self.parameters()) * 0
        loss_reg = torch.cat(reg).sum() / max(n_pos, 1) if reg else zero
        loss_q = torch.stack(qls).sum() / max(n_all, 1) if qls else zero
        return dict(loss_refine_reg=self.w_reg * loss_reg + zero,
                    loss_refine_q=self.w_q * loss_q + zero,
                    refine_pos=torch.tensor(float(n_pos), device=dev))
```

1. **匹配**：每个提议框和本帧所有 GT 算 BEV 旋转框 IoU，取最大的那个 GT（不限类别）。
2. **质量目标**（第 353–356 行）：只看**同类别** GT 的 IoU，映射为 `q* = clamp((IoU − 0.25)/0.5, 0, 1)`，即 IoU ≤ 0.25 时为 0，≥ 0.75 时为 1，中间线性。所有提议框都参与，损失是软标签的 BCE。空帧（没有 GT）时，所有提议框的质量目标都是 0。
3. **回归**（第 357–375 行）：IoU ≥ 0.25 的提议框为正样本，7 维目标：

| 目标 | 公式 | 回归它的分支 |
|---|---|---|
| Δρ | (c_gt − c)·u / 2 | `lidar_reg` |
| Δz | z_gt − z | `lidar_reg` |
| Δτ | (c_gt − c)·t / 1 | `fused_reg` |
| Δlog l,w,h | log(dim_gt / dim) | `fused_reg` |
| Δyaw | 差值折到 [−π/2, π/2) | `fused_reg` |

   朝向差按 π 折叠：框是左右对称的，船头和船尾对调的框在 IoU 上完全等价，不应受到惩罚。损失用 smooth-L1（β = 0.1），7 维求和后对正样本数取平均。

4. **DDP 保护**（第 379–383 行）：把 `0 × 所有参数` 加到损失里，即使整批都没有正样本，每个参数也都有梯度。

### 4.12 共用函数：图像分支、第二阶段的训练与推理

**[mari_fusion.py:391-399](maritime3d/mari_fusion.py#L391-L399)** · `image_branch`

```python
def image_branch(img_backbone, img_neck, imgs):
    """FPN features of every camera image, the per-camera brightness [B, N,
    2] (mean and std of the normalised band) and the image size."""
    img_hw = tuple(imgs.shape[-2:])
    x = imgs.flatten(0, 1) if imgs.dim() == 5 else imgs
    img_feats = img_neck(img_backbone(x))
    flat = imgs.flatten(2) if imgs.dim() == 5 else imgs[:, None].flatten(2)
    bright = torch.stack([flat.mean(-1), flat.std(-1)], -1)
    return img_feats, bright, img_hw
```

把 `[B, 2, 3, H, W]` 展平成 `[2B, 3, H, W]` 送进 ResNet 和 FPN，同时按相机算整条带的亮度均值和方差 `[B, 2, 2]`，供门控使用。

**[mari_fusion.py:402-412](maritime3d/mari_fusion.py#L402-L412)** · `refine_losses`

```python
def refine_losses(rh, dets, gts, bev, img_feats, metas, bright, img_hw):
    """Losses of the second stage on the stage-1 detections ``dets``."""
    props, scores, labels = rh.sample_props(dets, gts)
    r = rh(props, scores, labels, bev, img_feats, metas, bright, img_hw)
    losses = rh.loss(r, props, labels, gts)
    if img_feats is not None:
        # keep the image branch in the graph when no proposal projects into
        # a camera (DDP needs a gradient for every parameter)
        losses['loss_refine_q'] = losses['loss_refine_q'] + \
            sum(f.sum() for f in img_feats) * 0
    return losses
```

训练：采样提议框 → 前向 → 计算损失。和 `loss` 里的处理一样，这里也加上了 `0 × 图像特征之和`：如果某一步没有任何提议框投影进相机，图像分支的参数仍然会收到（零）梯度，DDP 不会报错。

**[mari_fusion.py:415-431](maritime3d/mari_fusion.py#L415-L431)** · `refine_predictions`

```python
def refine_predictions(rh, dets, bev, img_feats, metas, bright, img_hw):
    """Refined boxes and scores of the stage-1 detections ``dets``."""
    props = [d.bboxes_3d.tensor[:, :7] for d in dets]
    r = rh(props, [d.scores_3d for d in dets], [d.labels_3d for d in dets],
           bev, img_feats, metas, bright, img_hw)
    results = []
    for d, b, o, meta in zip(dets, props, r, metas):
        if o is None:
            results.append(d)
            continue
        out = InstanceData()
        out.bboxes_3d = meta['box_type_3d'](rh.refine_boxes(b.float(), o),
                                            box_dim=7)
        out.scores_3d = rh.score(d.scores_3d.float(), o)
        out.labels_3d = d.labels_3d
        results.append(out)
    return results
```

推理：第一阶段 NMS 后的**全部**检测都送进第二阶段（不截断到 128），用 `refine_boxes` 和 `score` 解码，类别保持不变。第二阶段之后**不再做 NMS**。

### 4.13 检测器：`MariFusionCenterPoint`

**[mari_fusion.py:434-486](maritime3d/mari_fusion.py#L434-L486)** · `MariFusionCenterPoint`

```python
@MODELS.register_module()
class MariFusionCenterPoint(CenterPoint):
    """CenterPoint + RaySplitFusionRefineHead (see module docstring)."""

    def __init__(self, *args, refine_head: dict = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refine_head = MODELS.build(refine_head)

    def _feats(self, inputs, metas):
        imgs = inputs.get('imgs', None)
        img_feats = bright = img_hw = None
        if imgs is not None and self.refine_head.use_image and \
                self.with_img_backbone:
            img_feats, bright, img_hw = image_branch(
                self.img_backbone, self.img_neck, imgs)
        pts_feats = self.extract_pts_feat(inputs.get('voxels'),
                                          points=inputs.get('points'),
                                          batch_input_metas=metas)
        return pts_feats, img_feats, bright, img_hw

    def _stage1(self, pts_feats, metas):
        outs = self.pts_bbox_head(pts_feats)
        det = [[{k: v.detach() for k, v in o[0].items()}] for o in outs]
        with torch.no_grad():
            dets = self.pts_bbox_head.predict_by_feat(det, metas)
        return outs, dets

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        gts = [s.gt_instances_3d for s in batch_data_samples]
        pts_feats, img_feats, bright, img_hw = self._feats(
            batch_inputs_dict, metas)
        outs, dets = self._stage1(pts_feats, metas)
        head = self.pts_bbox_head
        if hasattr(head, '_metas'):  # EvidenceSeaCenterHead: IMU tilt
            head._metas = metas
        try:
            losses = head.loss_by_feat(outs, gts)
        finally:
            if hasattr(head, '_metas'):
                head._metas = None
        losses.update(refine_losses(self.refine_head, dets, gts, pts_feats[0],
                                    img_feats, metas, bright, img_hw))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        pts_feats, img_feats, bright, img_hw = self._feats(
            batch_inputs_dict, metas)
        _, dets = self._stage1(pts_feats, metas)
        results = refine_predictions(self.refine_head, dets, pts_feats[0],
                                     img_feats, metas, bright, img_hw)
        return self.add_pred_to_datasample(batch_data_samples, results)
```

- `_feats`：图像分支和 LiDAR 分支分别提特征。`pts_feats[0]` 就是 B_L。
- `_stage1`：CenterHead 前向得到原始输出 `outs`（用来算第一阶段的损失）；再把 `outs` **detach** 之后走一遍 `predict_by_feat`（解码 + NMS），得到检测框 `dets` 作为第二阶段的输入。
- `loss`：第一阶段损失加第二阶段损失。如果第一阶段的头是带海面监督的 `EvidenceSeaCenterHead`，就把 meta（里面有 IMU 倾角）交给它，所以**最终模型在第二阶段训练时，海面监督仍然起作用**。
- `predict`：第一阶段 → 第二阶段 → 输出。

**梯度怎么流**：第二阶段的损失 → `RaySplitFusionRefineHead` → 通过 `grid_sample` 流回 **BEV 特征 B_L**（进而到 LiDAR 主干），以及流回**图像分支**。不会流回第一阶段的 CenterHead（框已 detach）。所以 LiDAR 主干同时受两个阶段的监督。

### 4.14 CenterFormer 版本：`MaritimeCenterFormer`

**[centerformer_maritime.py:136-219](maritime3d/centerformer_maritime.py#L136-L219)** · `MaritimeCenterFormer`

```python
@MODELS.register_module()
class MaritimeCenterFormer(CenterFormer):
    """CenterFormer for Maritime3D (see module docstring).

    Args:
        img_backbone, img_neck (dict, optional): Camera branch of the
            second stage.
        refine_head (dict, optional): RaySplitFusionRefineHead; None gives
            the one-stage detector.
    """

    def __init__(self,
                 *args,
                 img_backbone: dict = None,
                 img_neck: dict = None,
                 refine_head: dict = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_backbone = MODELS.build(img_backbone) \
            if img_backbone else None
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.refine_head = MODELS.build(refine_head) if refine_head else None
        # the BEV map the heatmap is predicted on (input of hm_head), for
        # the second stage
        self._bev = None
        self.backbone.hm_head.register_forward_pre_hook(self._grab_bev)

    def _grab_bev(self, module, inputs):
        if self.refine_head is not None:
            self._bev = inputs[0]

    def init_weights(self) -> None:
        super().init_weights()  # CenterFormer's: default init + BN weights
        for m in (self.img_backbone, self.img_neck, self.refine_head):
            if m is not None:
                m.init_weights()

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> Tensor:
        v = batch_inputs_dict['voxels']
        feats = self.voxel_encoder(v['voxels'], v['num_points'], v['coors'])
        batch_size = v['coors'][-1, 0].item() + 1
        return self.middle_encoder(feats, v['coors'], batch_size)

    def _images(self, batch_inputs_dict):
        imgs = batch_inputs_dict.get('imgs', None)
        if imgs is None or self.img_backbone is None or \
                self.refine_head is None or not self.refine_head.use_image:
            return None, None, None
        return image_branch(self.img_backbone, self.img_neck, imgs)

    def _stage1(self, batch_inputs_dict, batch_data_samples, metas):
        x = self.extract_feat(batch_inputs_dict, metas)
        preds, targets = self.backbone(x, batch_data_samples)
        bev, self._bev = self._bev, None
        return self.bbox_head(preds), targets, bev

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs) -> dict:
        metas = [s.metainfo for s in batch_data_samples]
        img_feats, bright, img_hw = self._images(batch_inputs_dict)
        preds, targets, bev = self._stage1(batch_inputs_dict,
                                           batch_data_samples, metas)
        losses = self.bbox_head.loss(preds, targets, metas=metas)
        if self.refine_head is not None:
            with torch.no_grad():
                # predict() permutes the tensors of its dicts in place
                det = [{k: v.detach() if torch.is_tensor(v) else v
                        for k, v in p.items()} for p in preds]
                dets = self.bbox_head.predict(det, metas)
            gts = [s.gt_instances_3d for s in batch_data_samples]
            losses.update(refine_losses(self.refine_head, dets, gts, bev,
                                        img_feats, metas, bright, img_hw))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        img_feats, bright, img_hw = self._images(batch_inputs_dict)
        preds, _, bev = self._stage1(batch_inputs_dict, batch_data_samples,
                                     metas)
        results = self.bbox_head.predict(preds, metas)
        if self.refine_head is not None:
            results = refine_predictions(self.refine_head, results, bev,
                                         img_feats, metas, bright, img_hw)
        return self.add_pred_to_datasample(batch_data_samples, results)
```

和 CenterPoint 版的区别只在"第二阶段从哪张 BEV 图采样"：

- CenterFormer 把 SECOND、FPN、heatmap 头和 transformer 都写在一个模块（`DeformableDecoderRPN`）里，外部拿不到中间的 BEV 图。这里在 `hm_head` 上注册了一个 **forward pre-hook**（第 161–165 行），在 heatmap 头被调用前把它的输入抓下来，即 **800×800、256 通道、0.4 m/格** 的 BEV 图；`_stage1` 取出后清空，避免下一帧误用。
- `extract_feat` 改成硬体素化，和 CenterPoint 用的 LiDAR 输入保持一致（原版 CenterFormer 用动态体素化）。
- 训练时第一阶段的输出要先 detach 再 `predict`，因为 CenterFormer 的 `predict` 会**原地** permute 字典里的张量，不复制就会改坏用于算损失的那份。
- 第二阶段完全复用 `image_branch / refine_losses / refine_predictions`。

### 4.15 对照基线：`BEVFuseCenterPoint`（BEV 级融合）

为什么不采用这种做法，见 1.1 节。

**[mari_fusion.py:489-552](maritime3d/mari_fusion.py#L489-L552)** · `BEVFuseCenterPoint`

```python
@MODELS.register_module()
class BEVFuseCenterPoint(CenterPoint):
    """CenterPoint with BEVFusion's camera branch (baseline for MariFusion).

    The images go through ``img_backbone`` / ``img_neck``, the LSS
    ``view_transform`` lifts them to a BEV grid and ``fusion_layer``
    (ConvFuser) concatenates it with the sparse encoder's BEV before the
    2D backbone, exactly as projects/BEVFusion does; the LiDAR path and the
    CenterHead are unchanged, so the LiDAR checkpoint loads as is.
    """

    def __init__(self, *args, view_transform: dict = None,
                 fusion_layer: dict = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.view_transform = MODELS.build(view_transform)
        self.fusion_layer = MODELS.build(fusion_layer)

    def init_weights(self) -> None:
        # as projects/BEVFusion: the LSS transform holds a Long parameter
        # that mmengine's generic init_weights cannot average
        self.img_backbone.init_weights()

    def extract_img_feat(self, img, input_metas):
        if img is None:
            return None
        B = img.shape[0]
        x = img.flatten(0, 1) if img.dim() == 5 else img
        x = self.img_neck(self.img_backbone(x))
        if not isinstance(x, torch.Tensor):
            x = x[0]
        return x.view(B, -1, *x.shape[1:])

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        # the image features are consumed inside the LiDAR path (fused
        # before the 2D backbone); MVXTwoStageDetector's loss / predict
        # would otherwise look for an image detection head
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'),
                                          batch_input_metas)
        pts_feats = self.extract_pts_feat(
            batch_inputs_dict['voxels'], points=batch_inputs_dict['points'],
            img_feats=img_feats, batch_input_metas=batch_input_metas)
        return None, pts_feats

    def extract_pts_feat(self, voxel_dict, points=None, img_feats=None,
                         batch_input_metas=None):
        vf = self.pts_voxel_encoder(voxel_dict['voxels'],
                                    voxel_dict['num_points'],
                                    voxel_dict['coors'])
        bs = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(vf, voxel_dict['coors'], bs)
        if img_feats is not None:
            def mat(key, default=None):
                rows = [m[key] if default is None else m.get(key, default)
                        for m in batch_input_metas]
                return x.new_tensor(np.asarray(rows, dtype=np.float32))
            with torch.autocast('cuda', enabled=False):
                img_bev = self.view_transform(
                    img_feats.float(), [p.float() for p in points],
                    mat('lidar2img'), mat('cam2img'), mat('cam2lidar'),
                    mat('img_aug_matrix', np.eye(4)),
                    mat('lidar_aug_matrix', np.eye(4)), batch_input_metas)
            x = self.fusion_layer([img_bev, x])
        x = self.pts_backbone(x)
        return self.pts_neck(x)
```

这是和 MariFusion 比较用的"同起点、同训练方式"的 BEV 级融合基线：图像经 Swin-T 编码，用 LSS 预测每个像素的深度分布，提升到 BEV（400×400，80 通道），再和 LiDAR 的 BEV 拼接，经 ConvFuser 融合后进入 2D 主干——完全照搬 BEVFusion 的做法，只是检测头换成 CenterPoint。

两者的本质区别：

| | BEV 级融合（LSS） | MariFusion（实例级） |
|---|---|---|
| 图像去哪 | 每个像素按深度分布撒到 BEV 网格 | 只在每个检测框的投影区域采样 |
| 视场外 3/4 的 BEV | 图像通道全是 0，仍参与卷积 | 门控为 0，严格等于纯 LiDAR |
| 海面倒影 | 倒影像素也会按深度撒到 BEV | 吃水线以下不采样 |
| 距离信息 | 依赖 LSS 从单目图像估深度（海面无纹理，很难估） | 距离只由 LiDAR 决定 |
| 参数量 / 推理时间 | 44.7 M / 162 ms | 20.2 M / 147 ms |

---

## 5. 用一帧真实数据走一遍几何

下面的数字来自 val 集第 485 帧中的一艘小艇（GT 框）。实际运行 `_project` 和 `_img_feat` 中的几何代码得到，不是估算。

| 量 | 数值 |
|---|---|
| GT 框 (x, y, z_底, l, w, h, yaw) | (88.83, −38.07, −1.82, 6.01, 1.96, 3.01, −0.49) |
| 距离 r | 96.6 m |
| 视线方向 u / 横向 t | (0.919, −0.394) / (0.394, 0.919) |
| 左相机：投影框 | u ∈ [1589.8, 1642.1]，v ∈ [362.0, 423.7]，约 52 × 62 px |
| 左相机：吃水线行 | v = 423.7 |
| 左相机：4×4 网格范围 | u ∈ [1579.3, 1652.6]，v ∈ [349.6, 425.7] ← 下边界 = 吃水线 + 2 px |
| 右相机：投影框 | u ∈ [1583.6, 1635.9]，v ∈ [345.6, 407.2] |

**灵敏度**，这就是 ray-split 的数据依据：

| 把框移动 1 m | 图像中心列的变化 | 吃水线行的变化 |
|---|---|---|
| 沿视线 | 0.07 px | 0.69 px |
| 横跨视线 | **20.7 px** | 0.07 px |

也就是说，在约 97 m 处，**相机分辨 1 m 的横向偏差非常容易（约 20 个像素），但分辨 1 m 的距离偏差几乎不可能（不到 1 个像素）**。换算一下，横向 1 px ≈ 4.8 cm，沿视线 1 px ≈ 1.45 m。所以距离和高度交给 LiDAR，横向、尺寸和朝向交给融合特征。

这艘艇在训练时的处理过程：
1. 它会有一个扰动副本：σ = max(6.01, 1.96)/10 = 0.6 m，所以沿视线偏移约 ±0.6 m、横向偏移约 ±0.3 m。
2. 网络读 5 个 BEV 点和两路相机各 16 个采样点；吃水线以下 2 px 之外的倒影不采。
3. 回归 Δρ ≈ 偏移量 / 2、Δτ ≈ 横向偏移 / 1，等等。

---

## 6. 训练与推理的区别

| | 训练 | 推理 |
|---|---|---|
| 第二阶段的输入框 | 第一阶段前 128 个检测 + GT 扰动副本 | 第一阶段 NMS 后的全部检测 |
| 门控 | 加模态 dropout | 无 dropout |
| 第一阶段损失 | 照常计算（含海面监督，如果有） | — |
| 第二阶段之后的 NMS | — | 不做 |
| 起点 | 加载第一阶段权重，再训 6 ep；图像骨干学习率 ×0.1 | — |

---

## 7. 实验结果：每个设计对应的证据

### 7.1 融合消融（同一起点：CenterPoint ep20，都再训 6 ep）

| 变体 | test mAP3D | test mAPBEV | Δ test 3D | 视场内 | 视场外 | yacht | val mAP3D |
|---|---|---|---|---|---|---|---|
| CenterPoint | 34.60 | 42.99 | — | 25.23 | 38.43 | 16.32 | 34.39 |
| + 第二阶段，不加相机 | 40.66 | 52.04 | +6.06 | 25.56 | 47.15 | 19.99 | 41.23 |
| + BEV 级融合（LSS） | 38.13 | 46.57 | +3.53 | 25.66 | 43.47 | 23.18 | 37.50 |
| **+ MariFusion** | **44.04** | **54.30** | **+9.44** | **30.33** | **49.79** | **35.25** | 40.72 |

怎么读：
- **"二阶段精修本身"贡献 +6.1**：同样的 ray-split 精修，不加相机也能涨很多，所以论文里必须把这一项单独列出来，才能说清楚相机到底贡献了多少。
- **相机的净贡献 = MariFusion − 不加相机 = +3.4**，并且集中在**视场内（+4.8）**，yacht 上 +15。这和"只有投影进相机的目标才拿到图像证据"的设计一致。
- **实例级比 BEV 级高 5.9**：BEV 级融合在视场内几乎没涨（25.2 → 25.7），说明 LSS 在海面上很难把图像有效地提升到 BEV。
- val 上相机反而略低（40.72 vs 41.23），差距主要来自 yacht（val 里只有 3 艘）。

### 7.2 最终模型（第一阶段训 80 ep）

| | test mAP3D | test mAPBEV | ATE-z | 视场内 | val mAP3D |
|---|---|---|---|---|---|
| CenterPoint 80ep | 48.14 | 56.07 | 0.72 | 31.60 | 46.06 |
| CenterPoint + 海面监督 80ep（第一阶段） | 49.24 | 58.18 | 0.71 | 29.95 | 42.53 |
| **+ MariFusion（最终模型）** | **49.43** | **60.90** | **0.70** | 30.94 | 44.40 |

**要注意：在充分训练（80 ep）的第一阶段上，MariFusion 的 3D 增益只有 +0.19，BEV 增益 +2.7；而在 20 ep 起点上，3D 增益是 +9.4。** 召回统计（1.1 节）支持第一种解释：80 ep 的第一阶段在视场内已经检测正确 67.7%（20 ep 是 58.8%），第二阶段的净修正从 +254 个降到 +131 个。另一种可能是第二阶段在 80 ep 起点上只训了 6 ep 还不够，两者并不互斥；要区分，需要补一个第二阶段训更久的实验。这一点需要在论文里如实说明。

### 7.3 复杂度

| | 参数量 | 推理 ms/帧（A800，FP32） |
|---|---|---|
| CenterPoint | 8.81 M | 105.7 |
| + 第二阶段，不加相机 | 9.80 M | 109.1 |
| MariFusion（CenterPoint） | 20.22 M（图像分支 10.12 M + 精修头 1.29 M） | 146.5 |
| CenterPoint + BEV 级融合 | 44.72 M | 162.3 |

---

## 8. 已知局限与代码注意点

1. **80 ep 起点上增益明显变小**（见 7.2），这是目前最需要解释的结果。
2. **门控用的亮度是整条带的统计量**，不是每个目标周围的局部亮度。逆光时某一侧过曝、但目标所在区域正常，这种情况区分不了。
3. **图像中目标之间的相互遮挡没有建模**：两条船在图像上重叠时，后面那条的采样网格会采到前面那条船。
4. **吃水线行取的是底面角点投影的最低行**，依赖第一阶段框的高度 z 和尺寸。如果第一阶段 z 偏差很大，"倒影屏蔽"会跟着偏（但上、左、右的外扩不受影响）。
5. **单帧**：没有用时序信息。
6. **没有在第二阶段后再做 NMS**：第二阶段修正后两个框可能重合，实际中很少见。
7. `self._last_gate` 没有被使用，是残留代码。
8. 第二阶段按帧循环、不做 batch 化。每帧实例数在几百以内，开销可以接受；如果要部署，可以改成 pad 后批处理。

---

## 9. 常见问题

**Q1：为什么不像 BEVFusion 那样把图像提升到 BEV？**
简单说：相机只覆盖约 1/4 视场，BEV 上 3/4 的格子没有图像；LSS 要从单目图像估深度，而海面几乎没有纹理；倒影会被一起提升到 BEV；受显存限制只能用低分辨率图像。同起点下，BEV 级融合比 MariFusion 低 5.9 mAP3D，视场内几乎没有提升。完整的论证、实验证据和这个选择的代价见 1.1 节。

**Q2：第二阶段会把梯度传回第一阶段吗？**
不会通过框传回（提议框 detach 了），但会通过 `grid_sample` 传回 BEV 特征和 LiDAR 主干。所以第一阶段的**检测头**只受第一阶段损失监督，**特征**同时受两个阶段监督。

**Q3：为什么距离和高度不让图像参与？**
第 5 节的实测：在 97 m 处，相机对 1 m 距离偏差的响应不到 1 px，信息量几乎为 0；如果让它参与回归，反而可能引入倒影、遮挡带来的噪声。高度同理，吃水线的像素位置主要由距离决定。

**Q4：模态 dropout 有什么用？**
一是让网络在相机失效（夜间、强光、镜头脏污）时能稳定退化为 LiDAR；二是防止网络偷懒，只看图像不看 LiDAR。训练时 25% 的帧完全没有相机，这些帧的输出由 `lidar_reg` 和 `fused_reg`（此时输入等于 `f_bev`）给出。

**Q5：为什么推理时第二阶段后不再做 NMS？**
第二阶段是逐框修正，修正幅度相对框本身较小，几乎不会让原本不重叠的两个框变得高度重叠；省掉这一步还能保证第二阶段只改框、不改检测数量，便于分析。
