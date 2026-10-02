# 物理海面监督（PSS）代码解析

版本：2026-10-02 · 适合读者：熟悉 PyTorch / mmdet3d，想逐行看懂海面模型的人。

| 文件 | 内容 |
|---|---|
| [maritime3d/sea_surface.py](maritime3d/sea_surface.py) | `SeaSurface`：IMU 倾角、Kalman 平均海平面、高斯过程波浪场、闭式联合求解 |
| [maritime3d/sea_up.py](maritime3d/sea_up.py) | `LoadSeaUp`：数据管线里读 IMU 重力方向（以及升沉） |
| [maritime3d/evidence_centerpoint.py](maritime3d/evidence_centerpoint.py) | `EvidenceSeaCenterHead`：把海面模型接进 CenterPoint 检测头，计算三个海面损失 |
| [maritime3d/centerformer_maritime.py](maritime3d/centerformer_maritime.py) | `SeaCenterFormerBboxHead`：CenterFormer 版本 |
| [configs/gn_centerpoint-ea-sea_maritime-bench.py](configs/gn_centerpoint-ea-sea_maritime-bench.py) | 海面模型的全部超参数 |
| [configs/gn_centerpoint-sea_maritime-bench.py](configs/gn_centerpoint-sea_maritime-bench.py) | 实际使用的配置（heatmap 峰在框中心）及数据管线 |

文中所有代码块都由 `work_dirs/tmp/docs/build_walkthroughs.py` 直接从源码抽出，行号和当前仓库一致。融合部分见 [MARIFUSION_CODE_WALKTHROUGH.md](MARIFUSION_CODE_WALKTHROUGH.md)。旧版的 SSR（刚性平面，推理时把框放到海面上，代码在 `wahead.py`）见 [SSR_CODE_WALKTHROUGH.md](SSR_CODE_WALKTHROUGH.md)，它已被本文的模型取代。

---

## 0. 一句话讲清楚

> **训练时，让每一帧里所有被检测到的船"一起"按物理规律估计出这一帧的海面，再要求网络根据这片海面推出的高度和真值一致。这个约束只在训练时用来教网络；推理时海面模型整个拿掉，不增加任何开销。**

打个比方：老师教学生估楼高。学生（检测头）一开始只会一栋一栋地猜。老师（海面模型）告诉他："这些楼都建在同一片坡地上，坡度可以用水平仪（IMU）量出来，相邻的楼地基高度差不多，大楼地基更稳。"学生按这个规律练习一段时间以后，即使不再看老师的提示，自己估得也更准了。

### 和旧版 SSR 的区别

| | 旧版 SSR（`wahead.py`） | 本文的 PSS |
|---|---|---|
| 海面形状 | 一帧一个刚性平面 | 平均海平面 + IMU 倾角 + **局部波浪**（高斯过程） |
| 大船和小船 | 一样对待 | 船越长，越"骑"在波浪上（按船长平均） |
| 时序 | 推理时用前几帧 | 一步 Kalman；训练时模拟先验（你选的方案 A） |
| 作用方式 | **推理时**把每个框放到海面上 | **只在训练时**作为监督，推理时不用 |
| 检测器 | TransFusion-L | CenterPoint、CenterFormer |

---

## 1. 物理模型

### 1.1 公式

增强后的 LiDAR 坐标系下，点 p = (x, y) 处的海面高度：

```
s(p) = tilt(p)          +    c          +    η(p)
       平均海面的倾斜          平均海平面高度       局部波浪
```

| 项 | 物理含义 | 怎么得到 | 参数 |
|---|---|---|---|
| `tilt(p) = (a·x + b·y)/100` | 平均海面和重力垂直，本船纵摇或横摇时，它在 LiDAR 坐标系里看起来是斜的 | IMU 给出重力方向 → (a, b)；纵摇、横摇两个轴各乘一个**可学习增益**，再加一个可学习的安装偏置 | 增益 `tilt_gain_delta`、偏置 `tilt_offset`；没有 IMU 的帧用 `prior_ab` |
| `c` | 本帧平均海平面在 LiDAR 坐标系里的高度 | 一步 Kalman：先验 (c⁻, P⁻) 加上本帧的观测，得到后验 (c⁺, P⁺) | 无历史时的先验 `prior_c`、`log_level_prior_var` |
| `η(p)` | 局部波浪，均值为 0 | 高斯过程，协方差见 1.3 | 波高 σ_w = `log_wave_sigma`，相关长度 ℓ = `log_wave_length` |

每个检测 j 都"观测"到自己的吃水线（底面高度）：

```
y_j = s(p_j) + e_j,      e_j ~ N(0, R_j),     R_j = 2·b_j² / score_j
```

b_j 由网络的 `sig` 分支预测（Laplace 尺度，方差为 2b²），再除以检测分数：网络越有把握、分数越高，这条观测就越可信。

### 1.2 数据依据（训练集统计）

- 同一帧里，各船底面相对这一帧共享海平面的偏差，中位数为 0.36 m：**海面不是一个刚性平面**。
- 两条船的偏差，相距 20 m 以内相关系数为 +0.52，50 m 以外接近 0：**波浪是局部的**，相关长度在几十米量级。
- 相邻帧（0.1 s）平均海平面的变化方差约为 4.9×10⁻³ m²，而 IMU 升沉只能解释其中不到 1%：所以 Kalman 的过程噪声取 `kalman_q = 0.005`，并且**不使用升沉**（`heave_gain = 0`）。

### 1.3 为什么"船越长越骑在浪上"

一条长 L 的船，它的吃水线是船身覆盖范围内海面的**平均值**。把波浪的高斯核沿船长做均匀平均，近似等于把核的方差加上 L²/12（长度为 L 的均匀分布，方差是 L²/12）。两条船 i、j 吃水线之间的协方差就变成：

```
k(p_i, p_j) = σ_w² · (ℓ² / ℓ_ij²) · exp(−|p_i − p_j|² / (2 ℓ_ij²)),     ℓ_ij² = ℓ² + (L_i² + L_j²)/12
```

- 对一条 70 m 的大船，ℓ_ii² 比 ℓ² 大很多，所以它的方差 σ_w²·ℓ²/ℓ_ii² 很小：**长船几乎不随浪起伏**；
- 两条 8 m 的小艇，ℓ_ij² ≈ ℓ²，就是原始的波浪相关性。

（严格说，船只沿船长方向平均，这里把它近似成各向同性的平均。）

---

## 2. 整体流程

### 2.1 训练时

```
点云 ──► LiDAR 主干 ──► BEV 特征 ──► CenterHead（每类一个 task）
                                      ├─ reg / height / dim / rot / a2c ──► 原有的框回归损失（不变）
                                      ├─ sig（新增，1 通道）：log b
                                      └─ fb （新增，1 通道）：freeboard 修正
                                               │
            ┌──────────────────────────────────┘  在每个 GT 中心所在的格子上读预测（teacher forcing）
            ▼
   观测：位置 x,y、船长 L、底面 bottom = z − h/2（都 detach）、log b、score（detach）、fb
            │
IMU（LoadSeaUp）─► tilt(a,b)               GT 底面 ─► GT 平均海平面 c_gt
            │                                    │
            │                       方案 A：50% 的概率模拟"有历史"：c⁻ = c_gt + 噪声
            ▼                                    ▼
      SeaSurface.solve：Kalman 更新 c  +  高斯过程克里金 → 每条船位置上的海面 ŝ_j，以及后验 c⁺
            │
            ├─ loss_sea_height = |ŝ_j + h_j/2 + fb_j − z_gt|      × 0.25
            ├─ loss_sigma      = |bottom_j − bottom_gt|/b_j + log b_j  × 0.25   （Laplace NLL）
            └─ loss_level      = |c⁺ − c_gt|                       × 0.5
```

### 2.2 推理时（`sea_at_test=False`，当前所有配置的默认值）

```
点云 ──► LiDAR 主干 ──► BEV 特征 ──► CenterHead ──► 框（高度 = height 分支自己的预测）
                                     （sig、fb 两个分支的输出被丢弃；SeaSurface 不运行）
```

"推理时也把框放到海面上"的版本保留为消融（`*-seatest_*` 配置），结果见第 9 节。

---

## 3. 输入：IMU 重力方向 `LoadSeaUp`

**[sea_up.py:1-117](maritime3d/sea_up.py#L1-L117)** · `whole file`

```python
"""Per-frame gravity direction for the IMU-anchored sea surface."""
import os
import warnings

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS

# Pack3DDetInputs' default meta keys plus the ones LoadSeaUp adds
_DEFAULT_META = Pack3DDetInputs.__init__.__defaults__[0]
SEA_UP_META_KEYS = tuple(_DEFAULT_META) + ('sea_up', 'sea_up_valid')
SEA_HEAVE_META_KEYS = SEA_UP_META_KEYS + ('sea_heave', 'sea_heave_valid',
                                          'sea_up_rot')


@TRANSFORMS.register_module()
class LoadSeaUp(BaseTransform):
    """Gravity up, in the (augmented) LiDAR frame, from a per-frame lookup.

    The LiDAR is body-fixed, so the sea tilts in its frame with the hull's
    attitude; the lookup holds the IMU-derived up vector for every frame,
    keyed by sequence and LiDAR timestamp (parsed from
    ``points4/<seq>/<ns>.bin``). Adds ``sea_up`` (float32 [3], unit) and
    ``sea_up_valid`` (bool); frames missing from the lookup, or flagged
    invalid, get (0, 0, 1) and ``sea_up_valid=False``. Pack them with
    ``Pack3DDetInputs(meta_keys=SEA_UP_META_KEYS)``.

    Place it after the augmentation: it applies ``lidar_aug_matrix`` (kept by
    the BEVFusion rotate / scale / translate / flip transforms), so the up
    vector follows the points. Other augmentations that do not record that
    matrix are refused rather than silently ignored.

    Args:
        lookup (str): npz with ``seq`` (str), ``ts`` (int64 ns), the up
            vectors (float [N, 3]) and ``valid`` (bool [N]).
        up_key (str): Which up vectors to use. dataset/infos/detection/
            sea_up_lidar.npz holds ``up_causal`` (past-only complementary
            filter, 5 s -- what an online system has, the default) and ``up``
            (zero-phase over +-10 s, an offline upper bound), both with the
            per-sequence mounting offset fitted on train+val horizons.
        heave_lookup (str, optional): npz with ``seq``, ``ts``, ``heave``
            (m, positive up, causal) and ``heave_valid``; adds ``sea_heave``
            and ``sea_heave_valid`` (pack with SEA_HEAVE_META_KEYS). Used by
            the sea-level Kalman prediction at test time only; heave is a
            relative displacement, so it is not touched by the augmentation.
        Always adds ``sea_up_rot`` (float32 [2, 2]): the map of the tilt from
        the body frame to the augmented frame (identity without
        augmentation), for the per-axis tilt gain of the sea surface.
    """

    def __init__(self, lookup: str, up_key: str = 'up_causal',
                 heave_lookup: str = None) -> None:
        self.lookup = lookup
        self.up_key = up_key
        self.heave_lookup = heave_lookup
        self._table = None  # loaded lazily, once per worker
        self._heave = None

    def _load(self):
        d = np.load(self.lookup, allow_pickle=False)
        up = d[self.up_key].astype(np.float64)
        valid = d['valid'].astype(bool)
        self._table = {(str(s), int(t)): (up[i], bool(valid[i]))
                       for i, (s, t) in enumerate(zip(d['seq'], d['ts']))}
        self._heave = {}
        if self.heave_lookup and not os.path.exists(self.heave_lookup):
            warnings.warn(f'{self.heave_lookup} not found: every frame gets '
                          'sea_heave_valid=False')
        elif self.heave_lookup:
            hv = np.load(self.heave_lookup, allow_pickle=False)
            self._heave = {
                (str(s), int(t)): (float(h), bool(v))
                for s, t, h, v in zip(hv['seq'], hv['ts'], hv['heave'],
                                      hv['heave_valid'])
            }

    def transform(self, results: dict) -> dict:
        if self._table is None:
            self._load()
        parts = str(results.get('lidar_path', '')).replace('\\',
                                                            '/').split('/')
        try:
            key = (parts[-2], int(parts[-1].split('.')[0]))
        except (IndexError, ValueError):
            key = None
        up, valid = self._table.get(key, (None, False))
        if up is None:
            up, valid = np.array([0.0, 0.0, 1.0]), False
        aug = results.get('lidar_aug_matrix')
        rot = np.eye(2)
        if aug is not None:
            m = np.asarray(aug, dtype=np.float64)[:3, :3]
            up = m @ up
            # tilt (a, b) = -s (up_x, up_y) / up_z maps from the body to the
            # augmented frame by m[:2, :2] / m[2, 2] (z rotation, BEV flips,
            # uniform scale): the frame in which a per-axis (pitch / roll)
            # correction has to be applied
            rot = m[:2, :2] / m[2, 2]
        elif any(k in results for k in ('pcd_rotation', 'pcd_horizontal_flip',
                                        'pcd_vertical_flip')):
            raise RuntimeError(
                'LoadSeaUp needs lidar_aug_matrix (BEVFusion augmentation '
                'transforms) to follow the augmentation of the points')
        results['sea_up'] = (up / np.linalg.norm(up)).astype(np.float32)
        results['sea_up_valid'] = bool(valid)
        results['sea_up_rot'] = rot.astype(np.float32)
        heave, hvalid = self._heave.get(key, (0.0, False))
        results['sea_heave'] = float(heave)
        results['sea_heave_valid'] = bool(hvalid)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(lookup={self.lookup!r}, '
                f'up_key={self.up_key!r}, '
                f'heave_lookup={self.heave_lookup!r})')
```

**它读什么**：
- `dataset/infos/detection/sea_up_lidar.npz`：每一帧（以序列号和 LiDAR 时间戳为键）在 LiDAR 坐标系下的重力"向上"单位向量。默认用 `up_causal`：只用过去 5 s 的 IMU 数据做互补滤波，也就是在线系统实际能拿到的那个值；另外 `up` 是 ±10 s 的零相位滤波，作为离线上界。两者都按序列减去了安装偏置（用 train+val 的地平线拟合得到）。
- `sea_heave_lidar.npz`：升沉。只有推理时的流式 Kalman 会用，当前配置没有用到。

**为什么要乘 `lidar_aug_matrix`**：训练时点云会被随机旋转（±45°）、缩放、翻转，海面的倾斜必须跟着点云一起变，否则"向上"方向就错了。

**`sea_up_rot` 是什么**：倾角 (a, b) = −100·(up_x, up_y)/up_z 是一个二维量。增强对它的作用，等于乘以 `m[:2, :2]/m[2, 2]`（绕 z 轴旋转、BEV 翻转、等比缩放）。纵摇和横摇的增益要在**船体坐标系**里分别施加，所以要记下这个映射，在 `tilt()` 里先把倾角映回船体坐标系、乘增益、再映回增强坐标系。

**输出到 meta**：

| key | 类型 | 含义 |
|---|---|---|
| `sea_up` | float32 [3] | 增强后坐标系里的重力向上单位向量 |
| `sea_up_valid` | bool | 这一帧是否有 IMU 数据 |
| `sea_up_rot` | float32 [2, 2] | 倾角从船体坐标系到增强坐标系的映射 |
| `sea_heave`、`sea_heave_valid` | float、bool | 升沉（当前不用） |

**在管线中的位置**（`gn_centerpoint-sea_maritime-bench.py`）：必须放在所有点云增强**之后**；同时 GT 框需要带 `anchors_3d`，这里用 `EvidenceAnchor(mode='centre')`，也就是锚点等于框中心（证据锚点已关闭）：

**[gn_centerpoint-sea_maritime-bench.py:1-8](configs/gn_centerpoint-sea_maritime-bench.py#L1-L8)** · `lines 1-8`

```python
"""Ablation: the physical sea surface on plain CenterPoint (heatmap at the
box centre; the anchor-to-centre vector is then zero).
Compare with gn_centerpoint-ea-sea_maritime-bench.py.
"""
_base_ = ['./gn_centerpoint-ea-sea_maritime-bench.py']

sea_up_lookup = 'dataset/infos/detection/sea_up_lidar.npz'
sea_heave_lookup = 'dataset/infos/detection/sea_heave_lidar.npz'
```

**[gn_centerpoint-sea_maritime-bench.py:54-92](configs/gn_centerpoint-sea_maritime-bench.py#L54-L92)** · `lines 54-92`

```python
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
```

**[evidence_centerpoint.py:36-78](maritime3d/evidence_centerpoint.py#L36-L78)** · `EvidenceAnchor`

```python
@TRANSFORMS.register_module()
class EvidenceAnchor(BaseTransform):
    """BEV anchor of every GT box: the centroid of its own LiDAR returns.

    Place after the augmentation and the last box filter, so the anchors are
    computed from the points and boxes the network sees. Adds
    ``gt_anchors_3d`` (float32 [N, 2]); pack it with
    :class:`Pack3DDetInputsAnchor`. Boxes without a return keep their centre.

    Args:
        mode (str): 'centroid' (evidence anchor) or 'centre' (plain
            CenterPoint; the anchor-to-centre vector is then zero).
        margin (float): Tolerance around the box in x / y (m).
        z_margin (float): Tolerance below the bottom / above the top (m).
    """

    def __init__(self, mode: str = 'centroid', margin: float = 0.3,
                 z_margin: float = 0.5) -> None:
        assert mode in ('centroid', 'centre')
        self.mode = mode
        self.margin = float(margin)
        self.z_margin = float(z_margin)

    def transform(self, results: dict) -> dict:
        t = results['gt_bboxes_3d'].tensor.numpy().astype(np.float64)
        anchors = t[:, :2].copy()
        if self.mode == 'centroid' and len(t):
            pts = results['points'].tensor.numpy()[:, :3].astype(np.float64)
            for i, (x, y, zb, l, w, h, yaw) in enumerate(t[:, :7]):
                c, s = np.cos(yaw), np.sin(yaw)
                dx, dy = pts[:, 0] - x, pts[:, 1] - y
                inside = (np.abs(dx * c + dy * s) <= l / 2 + self.margin) & \
                    (np.abs(-dx * s + dy * c) <= w / 2 + self.margin) & \
                    (pts[:, 2] >= zb - self.z_margin) & \
                    (pts[:, 2] <= zb + h + self.z_margin)
                if inside.any():
                    anchors[i] = pts[inside, :2].mean(0)
        results['gt_anchors_3d'] = torch.from_numpy(
            anchors.astype(np.float32))
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mode={self.mode!r})'
```

**[evidence_centerpoint.py:81-87](maritime3d/evidence_centerpoint.py#L81-L87)** · `Pack3DDetInputsAnchor`

```python
@TRANSFORMS.register_module()
class Pack3DDetInputsAnchor(Pack3DDetInputs):
    """Pack3DDetInputs that also packs ``gt_anchors_3d`` into
    ``gt_instances_3d.anchors_3d``."""
    INSTANCEDATA_3D_KEYS = Pack3DDetInputs.INSTANCEDATA_3D_KEYS + [
        'gt_anchors_3d'
    ]
```

`mode='centre'` 时 `EvidenceAnchor` 什么都不算，只是把框中心写成 `gt_anchors_3d`，让检测头的 `get_targets_single` 和证据锚点版本共用同一套代码；对应的 `a2c`（锚点到中心的向量）目标恒为 0。

---

## 4. 核心模块 `SeaSurface` 逐段解析

### 4.1 模块说明

**[sea_surface.py:1-36](maritime3d/sea_surface.py#L1-L36)** · `lines 1-36`

```python
"""Physical sea-surface model, shared by the maritime detection heads.

The sea height under a point p of the (augmented) LiDAR frame is

    s(p) = tilt(p) + c + eta(p),

* tilt(p) = (a x + b y) / s: the mean sea is normal to gravity, so its tilt
  comes from the IMU (``sea_up`` meta, see LoadSeaUp), scaled per body axis
  (pitch, roll) by a learned gain -- the annotated heights follow the hull's
  attitude only partly -- plus a learned mounting offset;
* c ~ N(c-, P-): the mean level, one Kalman step per frame. The prior is
  simulated in training (the GT level plus noise of random std s, P- = s^2)
  and streamed at test time (random walk with process noise ``kalman_q``
  per 0.1 s; IMU heave, measured to carry <1% of the annotated level's
  change, is optional via ``heave_gain``);
* eta: a zero-mean Gaussian-process wave field, k(p, q) = sw^2 l^2 / lij^2
  exp(-|p-q|^2 / 2 lij^2), lij^2 = l^2 + (L_p^2 + L_q^2) / 12: waves
  averaged over each hull's length L, so long ships ride over them. sw and l
  are learned. Box bottoms deviate from a frame's shared level by 0.36 m
  median and the deviations of two hulls correlate +0.52 within 20 m and ~0
  beyond 50 m (train split): local waves, not one rigid plane.

Detections observe their own waterline, y_j = s(p_j) + e_j with
e_j ~ N(0, 2 b_j^2 / score_j), b_j a learned Laplace scale. One closed-form
solve gives the Kalman update of c and the kriged sea height at every query:
a well-observed hull keeps its own waterline, a sparse far one borrows from
neighbours within ~l, an isolated one falls back to the mean sea. sw -> 0
recovers a rigid plane; no observations recover the Kalman prior.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

LOG_SIG_RANGE = (-3.0, 2.0)  # Laplace scale of a waterline: 5 cm .. 7.4 m
```

`LOG_SIG_RANGE = (−3, 2)`：网络预测的 log b 被限制在这个范围内，对应吃水线的不确定度在 5 cm 到 7.4 m 之间，防止数值爆炸。

### 4.2 构造函数：10 个可学习标量

**[sea_surface.py:58-91](maritime3d/sea_surface.py#L58-L91)** · `SeaSurface.__init__`

```python
    def __init__(self,
                 plane_scale: float = 100.0,
                 prior_c: float = -1.3,
                 level_prior_var: float = 1.0,
                 wave_sigma: float = 0.3,
                 wave_length: float = 25.0,
                 tilt_gain: bool = True,
                 max_obs: int = 50,
                 hist_prob: float = 0.5,
                 hist_std_range: Sequence[float] = (0.05, 1.0),
                 kalman_q: float = 0.005,
                 heave_gain: float = 0.0,
                 max_gap: float = 1.0) -> None:
        super().__init__()
        self.scale = float(plane_scale)
        self.max_obs = int(max_obs)
        self.hist_prob = float(hist_prob)
        self.hist_std_range = tuple(float(v) for v in hist_std_range)
        self.kalman_q = float(kalman_q)
        self.heave_gain = float(heave_gain)
        self.max_gap = float(max_gap)
        self.prior_c = nn.Parameter(torch.tensor(float(prior_c)))
        self.prior_ab = nn.Parameter(torch.zeros(2))  # frames without IMU
        self.tilt_offset = nn.Parameter(torch.zeros(2))
        # gain = 1 + delta, so weight decay pulls towards the IMU tilt
        self.tilt_gain_delta = nn.Parameter(torch.zeros(2)) \
            if tilt_gain else None
        self.log_level_prior_var = nn.Parameter(
            torch.tensor(float(level_prior_var)).log())
        self.log_wave_sigma = nn.Parameter(
            torch.tensor(float(wave_sigma)).log())
        self.log_wave_length = nn.Parameter(
            torch.tensor(float(wave_length)).log())
        self._stream = {}  # seq -> (timestamp_ns, c, P, heave or None)
```

**超参数**（不学习）：

| 参数 | 配置值 | 含义 |
|---|---|---|
| `plane_scale` | 100 | 倾角 (a, b) 的单位：a = 1 表示每 100 m 升高 1 m，让参数处于 O(1) 量级，便于优化 |
| `max_obs` | 50 | 每帧最多用 50 个最有把握的观测（求解是 O(n³)） |
| `hist_prob` | 0.5 | 训练时模拟"有历史"的概率 |
| `hist_std_range` | (0.05, 1.0) | 模拟历史的误差标准差，在这个范围内均匀采样（m） |
| `kalman_q` | 0.005 | 推理时流式 Kalman 的过程噪声（m²/0.1 s） |
| `heave_gain` | 0 | 推理时是否用升沉预测平均海平面的变化 |
| `max_gap` | 1.0 | 推理时相邻帧间隔超过 1 s 就重置流 |

**可学习参数**，以及在已训练模型里学到的值：

| 参数 | 维度 | 初值 | 20 ep 模型 | 80 ep 模型 | 最终模型（80+6） |
|---|---|---|---|---|---|
| `prior_c`：无历史时平均海平面的先验 | 1 | −1.3 m | −1.87 m | −2.86 m | −2.91 m |
| `log_level_prior_var` → 先验标准差 | 1 | 1.0 m | 1.56 m | 3.08 m | 3.23 m |
| `tilt_gain_delta` → 增益（纵摇，横摇） | 2 | (1, 1) | (0.62, 0.33) | **(−0.49, −0.27)** | **(−0.57, −0.30)** |
| `tilt_offset`：安装偏置 | 2 | 0 | ≈ 0° | ≈ 0.02° | ≈ 0.02° |
| `prior_ab`：无 IMU 帧的倾角 | 2 | 0 | ≈ 0 | ≈ 0 | ≈ 0 |
| `log_wave_sigma` → 波高 σ_w | 1 | 0.30 m | 0.40 m | 0.28 m | 0.29 m |
| `log_wave_length` → 相关长度 ℓ | 1 | 25 m | 22.7 m | 12.2 m | 12.4 m |

**80 ep 模型学到的倾角增益是负的，这在物理上说不通**，详见第 10 节。

几处设计说明：
- 增益写成 `1 + delta`：AdamW 的 weight decay 会把 delta 拉向 0，也就是默认"完全相信 IMU"。
- 方差、波高、相关长度都用 log 参数化，保证始终为正。
- `_stream` 是推理时流式 Kalman 的状态（每个序列存上一帧的 c、P 和时间戳），当前配置不使用。

### 4.3 辅助函数

**[sea_surface.py:94-96](maritime3d/sea_surface.py#L94-L96)** · `SeaSurface._keep`

```python
    def _keep(self) -> Tensor:
        """0 * every parameter: keeps them all in the graph (DDP)."""
        return sum(0 * p.sum() for p in self.parameters())
```

`0 × 每个参数之和`。加到输出上以后，数值不变，但每个参数都留在计算图里。DDP 要求每一步每个参数都有梯度；如果一个 batch 里所有帧都没有 IMU，`tilt_gain_delta` 就不会被用到，DDP 会直接报错（训练中真实出现过）。

**[sea_surface.py:98-105](maritime3d/sea_surface.py#L98-L105)** · `SeaSurface.stream_key`

```python
    @staticmethod
    def stream_key(meta: dict):
        """(seq, timestamp in ns) from points4/<seq>/<ns>.bin, or None."""
        parts = str(meta.get('lidar_path', '')).replace('\\', '/').split('/')
        try:
            return parts[-2], int(parts[-1].split('.')[0])
        except (IndexError, ValueError):
            return None
```

从 `points4/<序列>/<时间戳ns>.bin` 解析出（序列，时间戳），只用于推理时的流。

### 4.4 倾角 `tilt`

**[sea_surface.py:110-133](maritime3d/sea_surface.py#L110-L133)** · `SeaSurface.tilt`

```python
    def tilt(self, metas: List[dict], device) -> Tensor:
        """Mean-sea tilt (a, b) [B, 2] in the augmented LiDAR frame."""
        s = self.scale
        gain = None if self.tilt_gain_delta is None \
            else 1 + self.tilt_gain_delta
        rows = []
        for meta in metas:
            up = meta.get('sea_up')
            if up is None or not bool(meta.get('sea_up_valid', False)):
                rows.append(self.prior_ab)
                continue
            up = torch.as_tensor(up, dtype=torch.float32, device=device)
            ab = -s * up[:2] / up[2]
            if gain is not None:
                rot = meta.get('sea_up_rot')
                if rot is None:
                    assert not self.training, \
                        'the tilt gain needs the sea_up_rot meta in training'
                    rot = torch.eye(2)
                rot = torch.as_tensor(rot, dtype=torch.float32, device=device)
                # body frame -> scale pitch / roll -> augmented frame
                ab = rot @ (gain * torch.linalg.solve(rot, ab))
            rows.append(ab + self.tilt_offset)
        return torch.stack(rows) + self._keep()
```

对每一帧：

1. 没有 IMU，就用可学习的常数 `prior_ab`。
2. 重力向上方向 up = (u_x, u_y, u_z)。和它垂直的平面满足 u_x·x + u_y·y + u_z·z = 常数，即 z = −(u_x/u_z)·x − (u_y/u_z)·y + 常数，所以斜率 (a, b) = −100·(u_x, u_y)/u_z。
3. **分轴增益**：`ab = rot @ (gain * solve(rot, ab))`，即先用 `rot⁻¹` 映回船体坐标系，在那里纵摇（x 方向）乘 gain[0]、横摇（y 方向）乘 gain[1]，再用 `rot` 映回增强坐标系。为什么要在船体坐标系里乘：标注框的高度只是**部分地**跟随船体姿态，而纵摇和横摇被跟随的程度不同（见第 10 节的数据统计）。
4. 加上安装偏置 `tilt_offset`。

输出 `[B, 2]`。

**[sea_surface.py:135-137](maritime3d/sea_surface.py#L135-L137)** · `SeaSurface.tilt_z`

```python
    def tilt_z(self, tilt: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Tilt term (a x + b y) / s for x, y [B, n]."""
        return (tilt[:, 0:1] * x + tilt[:, 1:2] * y) / self.scale
```

在点 (x, y) 处的倾斜高度 (a·x + b·y)/100，输入输出都是 `[B, n]`。

### 4.5 GT 平均海平面 `gt_level`

**[sea_surface.py:139-147](maritime3d/sea_surface.py#L139-L147)** · `SeaSurface.gt_level`

```python
    def gt_level(self, xyz_bottom: Tensor, tilt: Tensor) -> Optional[Tensor]:
        """GT mean level of one frame: median of the box bottoms after
        removing the (detached) tilt; None without boxes."""
        if xyz_bottom.shape[0] == 0:
            return None
        t = tilt.detach().to(xyz_bottom)
        level = xyz_bottom[:, 2] - (t[0] * xyz_bottom[:, 0] +
                                    t[1] * xyz_bottom[:, 1]) / self.scale
        return level.median()
```

对一帧的所有 GT 框：底面高度减去该位置的倾斜高度（倾角先 detach），取**中位数**作为这一帧的真值平均海平面 c_gt。用中位数是为了不被个别标注异常的框带偏。没有 GT 框的帧返回 None。

### 4.6 Kalman 先验 `level_prior`（训练时用方案 A）

**[sea_surface.py:150-182](maritime3d/sea_surface.py#L150-L182)** · `SeaSurface.level_prior`

```python
    def level_prior(self, metas: List[dict],
                    gt_levels: Optional[List[Optional[Tensor]]] = None):
        """One-step Kalman prior (c-, P-) of the mean level, [B] each."""
        c0 = self.prior_c
        P0 = self.log_level_prior_var.exp()
        lo, hi = self.hist_std_range
        c, P = [], []
        for i, meta in enumerate(metas):
            ci, Pi = c0, P0
            if self.training:
                gt = None if gt_levels is None else gt_levels[i]
                if gt is not None and torch.rand(()) < self.hist_prob:
                    s = lo + (hi - lo) * float(torch.rand(()))
                    ci = gt.detach() + s * torch.randn((), device=c0.device)
                    Pi = c0.new_tensor(s * s)
            else:
                key = self.stream_key(meta)
                prev = self._stream.get(key[0]) if key else None
                if prev is not None:
                    ts, cp, Pp, hp = prev
                    dt = (key[1] - ts) * 1e-9
                    if 0 < dt <= self.max_gap:
                        h = meta.get('sea_heave')
                        dh = 0.0
                        if hp is not None and h is not None and \
                                bool(meta.get('sea_heave_valid', False)):
                            dh = float(h) - hp
                        ci = c0.new_tensor(cp - self.heave_gain * dh)
                        Pi = c0.new_tensor(Pp + self.kalman_q * dt / 0.1)
            c.append(ci)
            P.append(Pi)
        keep = self._keep()
        return torch.stack(c) + keep, torch.stack(P) + keep
```

**训练时**（`self.training = True`）：
- 以 50% 的概率模拟"上一帧已经把海平面估得差不多了"：随机取一个误差标准差 s ∈ [0.05, 1.0] m，令 c⁻ = c_gt + N(0, s²)，P⁻ = s²。这里 c_gt 是 detach 的，不回传梯度；
- 否则用"没有历史"的先验：c⁻ = `prior_c`，P⁻ = exp(`log_level_prior_var`)，这两个参数可学习。

这就是你选的**方案 A：训练时模拟先验**。好处是训练不需要按时间顺序取帧，可以照常随机打乱；同时网络在各种"先验好坏"的情况下都练过：先验很准时，观测的权重自动变小；先验很差时，观测的权重变大。

**推理时**（当前配置不走这条路）：取同一序列上一帧的后验 (c, P)，间隔 dt ≤ 1 s 就用 c⁻ = c_prev − heave_gain·Δheave、P⁻ = P_prev + q·dt/0.1；否则退回无历史先验。需要配合 `SequentialChunkSampler`，让每张卡按时间顺序处理连续的帧。

**[sea_surface.py:184-193](maritime3d/sea_surface.py#L184-L193)** · `SeaSurface.update_stream`

```python
    def update_stream(self, metas: List[dict], c: Tensor, P: Tensor) -> None:
        """Store this frame's posterior level (test time)."""
        for i, meta in enumerate(metas):
            key = self.stream_key(meta)
            if key is None:
                continue
            h = meta.get('sea_heave')
            ok = h is not None and bool(meta.get('sea_heave_valid', False))
            self._stream[key[0]] = (key[1], float(c[i]), float(P[i]),
                                    float(h) if ok else None)
```

推理时把本帧的后验存进流，供下一帧使用。

### 4.7 波浪协方差 `wave_cov`

**[sea_surface.py:196-203](maritime3d/sea_surface.py#L196-L203)** · `SeaSurface.wave_cov`

```python
    def wave_cov(self, xa, ya, La, xb, yb, Lb) -> Tensor:
        """Wave covariance [B, na, nb] between hull-averaged sea heights."""
        sw2 = (2 * self.log_wave_sigma).exp()
        l2 = (2 * self.log_wave_length).exp()
        lij2 = l2 + (La[:, :, None]**2 + Lb[:, None, :]**2) / 12.0
        d2 = (xa[:, :, None] - xb[:, None, :])**2 + \
            (ya[:, :, None] - yb[:, None, :])**2
        return sw2 * (l2 / lij2) * torch.exp(-0.5 * d2 / lij2)
```

就是 1.3 节的公式：输入两组位置和船长，输出 `[B, na, nb]` 的协方差矩阵。σ_w² 和 ℓ² 通过 exp 保证为正。

### 4.8 闭式联合求解 `solve`（核心）

**[sea_surface.py:205-242](maritime3d/sea_surface.py#L205-L242)** · `SeaSurface.solve`

```python
    def solve(self, tilt: Tensor, c_prior: Tensor, P_prior: Tensor,
              obs: Dict[str, Tensor], qry: Dict[str, Tensor]):
        """Sea height at the queries and the posterior mean level.

        obs: x, y, L (hull length), bottom (observed waterline), log_b,
            score and mask, [B, M] each; the positions, lengths and bottoms
            should be detached (only b, the tilt and the sea parameters
            learn through the fused height).
        qry: x, y, L, [B, Q] each.
        Returns sea [B, Q], c_post [B], P_post [B].
        """
        m = obs['mask'].float()
        if m.shape[1] > self.max_obs:  # the most confident observations
            keep = (obs['score'] * m).topk(self.max_obs, dim=1).indices
            obs = {k: v.gather(1, keep) for k, v in obs.items()}
            m = obs['mask'].float()
        x, y, L = obs['x'], obs['y'], obs['L']
        innov = (obs['bottom'] - self.tilt_z(tilt, x, y) -
                 c_prior[:, None]) * m
        log_b = obs['log_b'].clamp(*LOG_SIG_RANGE)
        R = 2 * (2 * log_b).exp() / obs['score'].clamp_min(1e-3)
        R = torch.where(m > 0, R, torch.full_like(R, 1e6))
        mm = m[:, :, None] * m[:, None, :]
        Pm = P_prior[:, None, None]
        S = Pm * mm + self.wave_cov(x, y, L, x, y, L) * mm + \
            torch.diag_embed(R)
        S = S + 1e-4 * torch.eye(S.shape[-1], device=S.device)
        rhs = torch.stack([innov, m], dim=-1)
        sol = torch.linalg.solve(S.double(), rhs.double()).to(S.dtype)
        alpha, s_inv_h = sol[..., 0], sol[..., 1]              # [B, M]
        c_post = c_prior + P_prior * (m * alpha).sum(1)
        P_post = P_prior - P_prior**2 * (m * s_inv_h).sum(1)
        # kriging: cov(s(q), y_O) = P- + k_wave(q, O), masked columns zero
        Kx = (Pm + self.wave_cov(qry['x'], qry['y'], qry['L'], x, y, L)) * \
            m[:, None, :]
        sea = self.tilt_z(tilt, qry['x'], qry['y']) + c_prior[:, None] + \
            (Kx @ alpha[..., None])[..., 0]
        return sea, c_post, P_post
```

**数学模型**：把一帧里所有观测写成一个向量 y（M 维）：

```
y = tilt(P_O) + 1·c + η_O + e
    c ~ N(c⁻, P⁻)          平均海平面（Kalman 先验）
    η_O ~ N(0, K)           波浪，K_ij = k(p_i, p_j)
    e ~ N(0, diag(R))       观测噪声
```

c、η、e 三者独立，所以 y 是高斯分布，均值 tilt + c⁻·1，协方差 **S = P⁻·11ᵀ + K + diag(R)**。对高斯分布做条件化，可以一次性得到所有想要的量：

| 量 | 公式 | 代码 |
|---|---|---|
| 新息 | v = y − tilt − c⁻ | `innov` |
| 求解 | α = S⁻¹ v | `alpha` |
| 平均海平面后验均值 | c⁺ = c⁻ + P⁻·1ᵀα | `c_post` |
| 平均海平面后验方差 | P⁺ = P⁻ − P⁻²·1ᵀS⁻¹1 | `P_post`（`s_inv_h = S⁻¹·m`） |
| 任意查询点 q 的海面（克里金） | ŝ(q) = tilt(q) + c⁻ + (P⁻·1 + k(q, O))ᵀ α | `sea` |

**逐行说明**：
1. **只用最有把握的 50 个观测**：按 `score × mask` 取 top-50。
2. **新息**：观测底面减去倾斜和先验海平面；无效位（padding）乘 mask 置零。
3. **观测噪声** R = 2b²/score：Laplace 分布的方差是 2b²；再除以分数，分数低的观测可信度降低。无效位的 R 设为 10⁶，相当于"无穷大噪声"，完全不起作用。
4. **组装 S**：先验项 P⁻·11ᵀ 和波浪项 K 都乘上 `mm = m·mᵀ`，把无效位的行和列清零；再加上 diag(R)。`1e-4·I` 是数值稳定项。
5. **一次求两个右端**：`rhs = [innov, m]`，同时解出 S⁻¹v（用于均值）和 S⁻¹m（用于方差），用 float64 求解以保证数值精度。
6. **Kalman 后验**：c⁺、P⁺。
7. **克里金**：查询点和观测点的协方差 = P⁻（共享的平均海平面）+ 波浪协方差；对无效观测的列置零。

**直观理解**（4.9 节有具体数字）：
- 一条观测很可靠的船（b 小、分数高）：R 很小，它的海面基本就是它自己的吃水线；
- 一条观测不可靠的船，如果附近有可靠的船：通过波浪协方差 K，向邻居"借"吃水线；
- 一条孤立且不可靠的船：K 接近 0，回落到平均海平面 c；
- 两个极限：σ_w → 0 时退化为一个刚性平面（tilt + c）；没有观测时 c⁺ = c⁻，即 Kalman 先验。这两个极限都在测试脚本里验证过。

### 4.9 用一个小例子跑一遍 `solve`

实际调用 `SeaSurface.solve` 得到的数字。倾角设为 0，波浪参数取初值（σ_w = 0.3 m，ℓ = 25 m）：

| 船 | 位置 (m) | 船长 | 观测底面 | b | 分数 | 观测噪声标准差 √R |
|---|---|---|---|---|---|---|
| A 小艇 | (30, 0) | 8 m | −1.10 | 0.1 | 0.9 | 0.15 m |
| B 小艇 | (45, 8) | 10 m | −1.25 | 0.2 | 0.8 | 0.32 m |
| C 大船 | (120, −30) | 70 m | −1.80 | 0.8 | 0.4 | 1.79 m |
| E（未观测，在 A 旁边） | (35, −5) | 6 m | — | | | |
| D（未观测，很远） | (150, 60) | 6 m | — | | | |

波浪协方差（m²）：A–A 0.088、A–B 0.070（离得近，强相关）、A–C 0.001（离得远，几乎不相关）、C–C 0.039（**70 m 的大船，自身的波浪方差只有小艇的一半不到**）。

求解结果：

| | ŝ(A) | ŝ(B) | ŝ(C) | ŝ(E) | ŝ(D) | c⁺ | √P⁺ |
|---|---|---|---|---|---|---|---|
| 无历史（c⁻ = −1.3，√P⁻ = 1.0） | −1.127 | −1.163 | −1.179 | −1.131 | −1.171 | −1.171 | 0.298 |
| 有历史（c⁻ = −1.20，√P⁻ = 0.05） | −1.131 | −1.171 | −1.207 | −1.137 | −1.199 | −1.199 | 0.049 |

读法：
- **A** 观测最可靠：海面 −1.127，基本保留了自己的 −1.10；
- **B** 观测到 −1.25，但它离 A 很近，A 更可靠，所以被拉向 A：−1.163（**向邻居借信息**）；
- **C** 观测到 −1.80，但噪声标准差 1.79 m，离其他船又远：海面 ≈ 平均海平面 −1.18（**不可靠就回落到均值**）；
- **E** 未被观测，在 A 旁边：跟着 A 走（−1.131）；**D** 未被观测，很远：等于 c⁺；
- **有历史**时，c⁻ 已经很准（±5 cm），远处的 C 和 D 就更多地相信先验。

### 4.10 训练时这些量怎么变成损失

`SeaSurface` 本身不计算损失，损失在检测头的 `_sea_losses` 里算，见第 5.6 节。

---

## 5. 接进 CenterPoint 检测头：`EvidenceSeaCenterHead`

### 5.1 构造函数：新增两个回归分支

**[evidence_centerpoint.py:211-236](maritime3d/evidence_centerpoint.py#L211-L236)** · `EvidenceSeaCenterHead.__init__`

```python
    def __init__(self,
                 *args,
                 a2c_scale: float = 2.0,
                 a2c_frame: str = 'bev',
                 sea_surface: dict = None,
                 loss_final_height_weight: float = 0.25,
                 loss_sigma_weight: float = 0.25,
                 loss_level_weight: float = 0.5,
                 sea_at_test: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        heads = self.task_heads[0].heads
        assert 'a2c' in heads, "needs common_heads['a2c'] = (2, 2)"
        assert a2c_frame in ('bev', 'local')
        self.a2c_scale = float(a2c_scale)
        self.a2c_frame = a2c_frame
        self.sea = None
        if sea_surface is not None:
            assert 'sig' in heads and 'fb' in heads, \
                "the sea surface needs common_heads 'sig' and 'fb'"
            self.sea = SeaSurface(**sea_surface)
        self.loss_final_height_weight = float(loss_final_height_weight)
        self.loss_sigma_weight = float(loss_sigma_weight)
        self.loss_level_weight = float(loss_level_weight)
        self.sea_at_test = bool(sea_at_test)
        self._metas = None
```

- 配置里加了 `common_heads=dict(sig=(1, 2), fb=(1, 2))`：每个类别的 task 多出两个分支，各是"3×3 卷积 64 通道 → 1 通道"，和 reg、height 等分支并列，共享 `shared_conv` 和 BEV 主干。
  - `sig`：log b，吃水线的不确定度；
  - `fb`：freeboard 修正，表示"海面推出来的中心高度"还差多少。
- `a2c` 分支来自证据锚点版本；`mode='centre'` 时它的目标恒为 0，是约 0.19 M 的冗余参数，保留是为了和已有权重兼容。
- `sea_at_test=False`（配置值）：推理时不用海面。

### 5.2 训练目标 `get_targets_single`

**[evidence_centerpoint.py:239-300](maritime3d/evidence_centerpoint.py#L239-L300)** · `EvidenceSeaCenterHead.get_targets_single`

```python
    def get_targets_single(self, gt_instances_3d):
        """CenterHead targets with the heatmap peak at the evidence anchor.

        Returns per task: heatmap [n_cls, H, W], code [max_objs, 10],
        anchor-cell index [max_objs] and mask [max_objs].
        """
        labels = gt_instances_3d.labels_3d
        device = labels.device
        b3d = gt_instances_3d.bboxes_3d
        boxes = torch.cat((b3d.gravity_center, b3d.tensor[:, 3:7]),
                          dim=1).to(device)
        if 'anchors_3d' in gt_instances_3d:
            anchors = gt_instances_3d.anchors_3d.to(device).float()
        else:
            anchors = boxes[:, :2]
        cfg = self.train_cfg
        max_objs = cfg['max_objs'] * cfg['dense_reg']
        pcr, vs, osf = cfg['point_cloud_range'], cfg['voxel_size'], \
            cfg['out_size_factor']
        fw = int(cfg['grid_size'][0]) // osf
        fh = int(cfg['grid_size'][1]) // osf
        heatmaps, codes, inds, masks = [], [], [], []
        flag = 0
        for names in self.class_names:
            heatmap = boxes.new_zeros(len(names), fh, fw)
            code = boxes.new_zeros(max_objs, 10)
            ind = labels.new_zeros(max_objs, dtype=torch.int64)
            mask = boxes.new_zeros(max_objs, dtype=torch.uint8)
            sel = torch.cat([
                torch.where(labels == flag + j)[0] for j in range(len(names))
            ])
            cls_ids = labels[sel] - flag
            flag += len(names)
            for k in range(min(len(sel), max_objs)):
                b, a = boxes[sel[k]], anchors[sel[k]]
                length = b[3] / vs[0] / osf
                width = b[4] / vs[1] / osf
                if not (width > 0 and length > 0):
                    continue
                radius = gaussian_radius(
                    (width, length), min_overlap=cfg['gaussian_overlap'])
                radius = max(cfg['min_radius'], int(radius))
                axy = torch.stack([(a[0] - pcr[0]) / vs[0] / osf,
                                   (a[1] - pcr[1]) / vs[1] / osf])
                aint = axy.to(torch.int32)
                if not (0 <= aint[0] < fw and 0 <= aint[1] < fh):
                    continue
                draw_heatmap_gaussian(heatmap[cls_ids[k]], aint, radius)
                ind[k] = aint[1] * fw + aint[0]
                mask[k] = 1
                dim = b[3:6].log() if self.norm_bbox else b[3:6]
                code[k] = torch.cat([
                    axy - aint.float(), b[2:3], dim,
                    torch.sin(b[6:7]), torch.cos(b[6:7]),
                    a2c_code(b[:2] - a, b[3:5], b[6], self.a2c_frame) /
                    self.a2c_scale
                ])
            heatmaps.append(heatmap)
            codes.append(code)
            inds.append(ind)
            masks.append(mask)
        return heatmaps, codes, inds, masks
```

和原版 CenterHead 基本一样：heatmap 上画高斯峰；回归目标是 10 维 code，即 [格内偏移 (2)、**重心** z (1)、log l,w,h (3)、sin yaw、cos yaw、a2c (2)]。`mode='centre'` 时锚点等于框中心，所以格内偏移就是普通 CenterPoint 的偏移，a2c 为 0。

**海面损失会用到的两个量**：`ind`（GT 中心所在格子的展平索引）和 `code`（GT 的回归目标）。

### 5.3 把 meta 交给海面模型 `loss`

**[evidence_centerpoint.py:303-310](maritime3d/evidence_centerpoint.py#L303-L310)** · `EvidenceSeaCenterHead.loss`

```python
    def loss(self, pts_feats, batch_data_samples, *args, **kwargs):
        """Keeps the metas at hand: the sea surface needs the IMU tilt."""
        self._metas = [s.metainfo for s in batch_data_samples]
        try:
            return super().loss(pts_feats, batch_data_samples, *args,
                                **kwargs)
        finally:
            self._metas = None
```

mmdet3d 的 `CenterHead.loss` 只把 GT 实例传给 `loss_by_feat`，不传 meta。海面模型需要 meta 里的 IMU 倾角，所以先把 meta 暂存到 `self._metas`，算完后在 `finally` 里清掉。

### 5.4 读取 GT 格子上的预测 `loss_by_feat`

**[evidence_centerpoint.py:312-315](maritime3d/evidence_centerpoint.py#L312-L315)** · `EvidenceSeaCenterHead._gather`

```python
    def _gather(self, x: Tensor, ind: Tensor) -> Tensor:
        """[B, C, H, W] map at [B, K] flat indices -> [B, K, C]."""
        x = x.permute(0, 2, 3, 1).contiguous()
        return self._gather_feat(x.view(x.size(0), -1, x.size(3)), ind)
```

**[evidence_centerpoint.py:317-349](maritime3d/evidence_centerpoint.py#L317-L349)** · `EvidenceSeaCenterHead.loss_by_feat`

```python
    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args,
                     **kwargs):
        heatmaps, codes, inds, masks = self.get_targets(batch_gt_instances_3d)
        loss_dict = dict()
        at_gt = []
        for task_id, preds_dict in enumerate(preds_dicts):
            p = preds_dict[0]
            p['heatmap'] = clip_sigmoid(p['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_dict[f'task{task_id}.loss_heatmap'] = self.loss_cls(
                p['heatmap'], heatmaps[task_id], avg_factor=max(num_pos, 1))
            target = codes[task_id]
            pred = self._gather(
                torch.cat([p[k] for k in ('reg', 'height', 'dim', 'rot',
                                          'a2c')], dim=1), inds[task_id])
            num = masks[task_id].float().sum()
            mask = masks[task_id].unsqueeze(2).expand_as(target).float()
            mask *= (~torch.isnan(target)).float()
            weights = mask * mask.new_tensor(self.train_cfg['code_weights'])
            loss_dict[f'task{task_id}.loss_bbox'] = self.loss_bbox(
                pred, target, weights, avg_factor=(num + 1e-4))
            if self.sea is not None:
                score = self._gather(p['heatmap'].detach(),
                                     inds[task_id]).max(-1).values
                at_gt.append(dict(
                    pred=pred, tgt=target, ind=inds[task_id],
                    mask=masks[task_id].float(), score=score,
                    log_b=self._gather(p['sig'], inds[task_id])[..., 0],
                    fb=self._gather(p['fb'], inds[task_id])[..., 0]))
        if self.sea is not None:
            with torch.autocast('cuda', enabled=False):
                loss_dict.update(self._sea_losses(at_gt))
        return loss_dict
```

前半部分（heatmap focal loss 和 L1 框回归）就是原版 CenterHead。新增的是 `if self.sea is not None` 那一段：对每个 task，**在 GT 中心所在的格子上**收集：

| 量 | 来源 | 形状 |
|---|---|---|
| `pred` | 网络在这些格子上的 10 维回归输出 | `[B, max_objs, 10]` |
| `tgt` | 对应的 GT 回归目标 | `[B, max_objs, 10]` |
| `ind`、`mask` | 格子索引，以及这个位置是否真的有 GT | `[B, max_objs]` |
| `score` | 这个格子的 heatmap 分数（**detach**） | `[B, max_objs]` |
| `log_b`、`fb` | 新增两个分支的输出 | `[B, max_objs]` |

这就是 **teacher forcing**：训练时不用网络自己的检测结果（可能漏检、可能偏），而是直接在"应该有船"的位置上读它的预测，保证每个 GT 都参与海面估计。5 个类别各是一个 task，最后在 `_sea_losses` 里拼到一起，所以**不同类别的船共享同一片海面**。

### 5.5 格子索引 → 米制坐标 `_cell_to_m`

**[evidence_centerpoint.py:351-367](maritime3d/evidence_centerpoint.py#L351-L367)** · `EvidenceSeaCenterHead._cell_to_m`

```python
    def _cell_to_m(self, ind: Tensor, code: Tensor):
        """Box centre (m) from the anchor cell index and the box code
        [sub-cell offset (2), z, log l w h (3), sin, cos, a2c (2)]."""
        cfg = self.train_cfg
        osf, vs, pcr = cfg['out_size_factor'], cfg['voxel_size'], \
            cfg['point_cloud_range']
        fw = int(cfg['grid_size'][0]) // osf
        cx = (ind % fw).float() + code[..., 0]
        cy = (ind // fw).float() + code[..., 1]
        dim = code[..., 3:6].clamp(*LOG_DIM_RANGE).exp() if self.norm_bbox \
            else code[..., 3:6]
        yaw = torch.atan2(code[..., 6], code[..., 7])
        off = a2c_offset(code[..., 8:10] * self.a2c_scale, dim, yaw,
                         self.a2c_frame)
        x = cx * vs[0] * osf + pcr[0] + off[..., 0]
        y = cy * vs[1] * osf + pcr[1] + off[..., 1]
        return x, y
```

中心 = (格子列号 + 格内偏移) × 0.8 m + 范围起点 + 锚点到中心的向量（这里为 0）。输入可以是网络预测的 code，也可以是 GT 的 code。

### 5.6 三个海面损失 `_sea_losses`（核心）

**[evidence_centerpoint.py:369-418](maritime3d/evidence_centerpoint.py#L369-L418)** · `EvidenceSeaCenterHead._sea_losses`

```python
    def _sea_losses(self, at_gt: List[dict]) -> dict:
        """Sea surface on the predictions at the GT anchor cells."""
        cat = {k: torch.cat([g[k] for g in at_gt], dim=1) for k in at_gt[0]}
        valid = cat['mask'] > 0
        # compact: the GT objects of every sample first, padded to n >= 1
        n = max(1, int(valid.sum(1).max()))
        order = torch.sort(valid.float(), dim=1, descending=True,
                           stable=True).indices[:, :n]
        g = {}
        for k, v in cat.items():
            idx = order if v.dim() == 2 else \
                order[..., None].expand(-1, -1, v.shape[-1])
            g[k] = v.gather(1, idx)
        m = g['mask']
        pred, tgt = g['pred'].float(), g['tgt']
        logd = pred[..., 3:6].clamp(*LOG_DIM_RANGE).detach()
        L, h = logd[..., 0].exp(), logd[..., 2].exp()
        x, y = self._cell_to_m(g['ind'], pred.detach())
        bottom = pred[..., 2].detach() - 0.5 * h
        z_gt = tgt[..., 2]
        bottom_gt = z_gt - 0.5 * tgt[..., 5].exp()
        gx, gy = self._cell_to_m(g['ind'], tgt)
        metas = self._metas or [{} for _ in range(m.shape[0])]
        tilt = self.sea.tilt(metas, m.device)
        levels = []
        for i in range(m.shape[0]):
            v = m[i] > 0
            levels.append(self.sea.gt_level(
                torch.stack([gx[i][v], gy[i][v], bottom_gt[i][v]], dim=1),
                tilt[i]))
        c_prior, P_prior = self.sea.level_prior(metas, levels)
        sea, c_post, _ = self.sea.solve(
            tilt, c_prior, P_prior,
            obs=dict(x=x, y=y, L=L, bottom=bottom, log_b=g['log_b'].float(),
                     score=g['score'].float(), mask=m),
            qry=dict(x=x, y=y, L=L))
        z = sea + 0.5 * h + g['fb'].float()
        denom = m.sum().clamp_min(1.0)
        losses = dict(
            loss_sea_height=self.loss_final_height_weight *
            ((z - z_gt).abs() * m).sum() / denom)
        err = (bottom - bottom_gt).abs().detach()
        log_b = g['log_b'].float().clamp(*LOG_SIG_RANGE)
        losses['loss_sigma'] = self.loss_sigma_weight * (
            (err * (-log_b).exp() + log_b) * m).sum() / denom
        lv = [(c_post[i] - gt).abs() for i, gt in enumerate(levels)
              if gt is not None]
        losses['loss_level'] = self.loss_level_weight * (
            torch.stack(lv).mean() if lv else 0 * c_post.sum())
        return losses
```

逐步说明：

1. **拼接并压紧**：把 5 个 task 的观测拼成 `[B, 5·max_objs]`，再按 mask 排序，把每帧真实的 GT 排到前面，截到这一批中 GT 数最多的那一帧的数量 n（至少 1）。这样矩阵求解的规模是 n 而不是 2500。
2. **取观测**（这一步全部 detach）：
   - 船长 L、高度 h：来自网络预测的 log 尺寸；
   - 位置 (x, y)：网络预测的格内偏移；
   - 观测底面 `bottom = 预测的重心 z − h/2`。
3. **GT 量**：GT 重心 z、GT 底面、GT 位置 (gx, gy)。
4. **倾角**：`self.sea.tilt(metas)`。
5. **每帧 GT 平均海平面** c_gt：`gt_level`，用的是 GT 位置和 GT 底面。
6. **Kalman 先验**：`level_prior(metas, levels)`，即方案 A。
7. **求解**：`solve`，**查询点就是观测点本身**，得到每条船位置上的海面 ŝ 和后验 c⁺。
8. **三个损失**（都只在有效的 GT 位置上平均）：

| 损失 | 公式 | 权重 | 含义 |
|---|---|---|---|
| `loss_sea_height` | \|ŝ + h/2 + fb − z_gt\| | 0.25 | 海面加半个船高加 freeboard 修正，应该等于 GT 重心高度 |
| `loss_sigma` | \|bottom − bottom_gt\|/b + log b | 0.25 | Laplace 负对数似然：要求 b 真实反映"这条船自己的吃水线估得有多准"（误差 detach） |
| `loss_level` | \|c⁺ − c_gt\| | 0.5 | 后验平均海平面应该接近真值 |

### 5.7 梯度到底流向哪里

这一点决定了海面监督"怎么起作用"，必须说清楚：

| 量 | 是否回传梯度 | 原因 |
|---|---|---|
| 预测的位置 x, y、尺寸 L, h、自身高度（bottom） | **否**（detach） | 防止海面损失直接去改框；框由原有的回归损失负责 |
| 分数 score | 否 | 同上 |
| `sig` 分支（log b） | **是**：来自三个损失（经过 R 和 Laplace NLL） | |
| `fb` 分支 | **是**：来自 `loss_sea_height` | |
| `SeaSurface` 的 10 个参数 | **是** | |
| `shared_conv` 和 BEV 主干 | **是**：经由 sig 和 fb 两个分支 | 海面监督正是通过这条路径影响检测 |
| `height` 分支 | **否**，只受原有 L1 损失监督 | |

所以，**海面监督是一个物理约束的辅助任务**：它不直接改框的高度，而是要求共享特征同时能支持"判断自己的吃水线有多可靠"（sig）和"相对海面的干舷"（fb）。这两件事都需要特征里编码目标和周围海面的几何关系，这种更好的特征也让 `height` 分支估得更准（80 ep：ATE-z 从 0.72 降到 0.71；15–30 m 的船从 0.69 降到 0.65）。

### 5.8 推理 `predict_by_feat` 与消融用的 `_sea_bottoms`

**[evidence_centerpoint.py:421-437](maritime3d/evidence_centerpoint.py#L421-L437)** · `EvidenceSeaCenterHead.predict_by_feat`

```python
    def predict_by_feat(self, preds_dicts, batch_input_metas, *args,
                        **kwargs):
        for preds_dict in preds_dicts:
            p = preds_dict[0]
            extra = [p['sig'], p['fb']] if self.sea is not None else \
                [torch.zeros_like(p['a2c'])]
            # decoded by EvidenceCenterPointBBoxCoder from the 'vel' slot
            p['vel'] = torch.cat([p['a2c']] + extra, dim=1)
        results = super().predict_by_feat(preds_dicts, batch_input_metas,
                                          *args, **kwargs)
        for meta, r in zip(batch_input_metas, results):
            b = r.bboxes_3d.tensor  # z is the bottom, cols 7/8: log_b, fb
            box = b[:, :7].clone()
            if self.sea is not None and self.sea_at_test:
                box[:, 2] = self._sea_bottoms(meta, b, r.scores_3d)
            r.bboxes_3d = meta['box_type_3d'](box, box_dim=7)
        return results
```

借用 CenterPoint 解码器的 `vel` 通道，把 [a2c、log b、fb] 一起解码出来（解码器见下），然后：
- `sea_at_test=False`（默认）：框的 z 就是 `height` 分支的预测，只取前 7 维；
- `sea_at_test=True`（消融）：调用 `_sea_bottoms`，把框的底面换成海面 + fb。

**[evidence_centerpoint.py:439-458](maritime3d/evidence_centerpoint.py#L439-L458)** · `EvidenceSeaCenterHead._sea_bottoms`

```python
    def _sea_bottoms(self, meta: dict, b: Tensor, scores: Tensor) -> Tensor:
        """Bottoms of one frame's detections on the sea surface."""
        with torch.autocast('cuda', enabled=False):
            b = b.float()
            tilt = self.sea.tilt([meta], b.device)
            c_prior, P_prior = self.sea.level_prior([meta])
            if b.shape[0] == 0:
                if not self.training:
                    self.sea.update_stream([meta], c_prior, P_prior)
                return b[:, 2]
            x, y, L = b[None, :, 0], b[None, :, 1], b[None, :, 3]
            sea, c_post, P_post = self.sea.solve(
                tilt, c_prior, P_prior,
                obs=dict(x=x, y=y, L=L, bottom=b[None, :, 2],
                         log_b=b[None, :, 7], score=scores[None].float(),
                         mask=torch.ones_like(x)),
                qry=dict(x=x, y=y, L=L))
            if not self.training:  # stage-1 proposals of MariFusion
                self.sea.update_stream([meta], c_post, P_post)
            return sea[0] + b[:, 8]
```

消融路径：NMS 之后的检测框作为观测，求解后每个框的底面 = 海面 + fb，并把后验存进流。

**[evidence_centerpoint.py:133-169](maritime3d/evidence_centerpoint.py#L133-L169)** · `EvidenceCenterPointBBoxCoder.decode`

```python
    def decode(self, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None,
               task_id=-1) -> List[dict]:
        batch = heat.shape[0]
        K = self.max_num
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        def gather(x, c):
            return self._transpose_and_gather_feat(x, inds).view(batch, K, c)

        xs, ys = xs.view(batch, K, 1), ys.view(batch, K, 1)
        if reg is not None:
            reg = gather(reg, 2)
            xs, ys = xs + reg[..., 0:1], ys + reg[..., 1:2]
        else:
            xs, ys = xs + 0.5, ys + 0.5
        extra = gather(vel, 4)
        cell_x = self.out_size_factor * self.voxel_size[0]
        cell_y = self.out_size_factor * self.voxel_size[1]
        rot = torch.atan2(gather(rot_sine, 1), gather(rot_cosine, 1))
        dim = gather(dim, 3)
        off = a2c_offset(extra[..., 0:2] * self.a2c_scale, dim, rot,
                         self.a2c_frame)
        xs = xs * cell_x + self.pc_range[0] + off[..., 0:1]
        ys = ys * cell_y + self.pc_range[1] + off[..., 1:2]
        boxes = torch.cat(
            [xs, ys, gather(hei, 1), dim, rot, extra[..., 2:4]], dim=2)
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)
        pcr = torch.as_tensor(self.post_center_range, device=heat.device)
        mask = (boxes[..., :3] >= pcr[:3]).all(2)
        mask &= (boxes[..., :3] <= pcr[3:]).all(2)
        if self.score_threshold:
            mask &= scores > self.score_threshold
        return [
            dict(bboxes=boxes[i, mask[i]], scores=scores[i, mask[i]],
                 labels=clses[i, mask[i]]) for i in range(batch)
        ]
```

解码器：标准的 CenterPoint top-K 解码，额外从 `vel` 通道取出 4 个数（a2c 2 维、log b、fb），输出 9 维框。

---

## 6. 接进 CenterFormer：`SeaCenterFormerBboxHead`

**[centerformer_maritime.py:35-100](maritime3d/centerformer_maritime.py#L35-L100)** · `lines 35-100`

```python
@MODELS.register_module()
class SeaCenterFormerBboxHead(CenterFormerBboxHead):
    """CenterFormerBboxHead + sea-surface supervision (see module docstring).

    Args:
        train_cfg (dict): The detector's train_cfg (grid, voxel size, ...).
        sea_surface (dict, optional): :class:`SeaSurface` arguments; None
            gives the plain head (with the generalised NMS).
        loss_final_height_weight, loss_sigma_weight, loss_level_weight
            (float): As :class:`EvidenceSeaCenterHead`.
    """

    # the sea-surface losses of the CenterPoint head; they read the box code
    # [sub-cell offset (2), z, log l w h (3), sin, cos, a2c (2)] at flat
    # heatmap indices, which is CenterFormer's code plus a zero a2c
    _cell_to_m = EvidenceSeaCenterHead._cell_to_m
    _sea_losses = EvidenceSeaCenterHead._sea_losses

    def __init__(self,
                 *args,
                 train_cfg: dict = None,
                 sea_surface: dict = None,
                 loss_final_height_weight: float = 0.25,
                 loss_sigma_weight: float = 0.25,
                 loss_level_weight: float = 0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_cfg = train_cfg
        self.norm_bbox = True  # CenterFormer regresses log dimensions
        self.a2c_scale, self.a2c_frame = 1.0, 'bev'
        self.sea = None
        if sea_surface is not None:
            heads = self.tasks[0].heads
            assert 'sig' in heads and 'fb' in heads, \
                "the sea surface needs common_heads 'sig' and 'fb'"
            self.sea = SeaSurface(**sea_surface)
        self.loss_final_height_weight = float(loss_final_height_weight)
        self.loss_sigma_weight = float(loss_sigma_weight)
        self.loss_level_weight = float(loss_level_weight)
        self._metas = None

    def loss(self, preds_dicts, example, metas: List[dict] = None,
             **kwargs) -> dict:
        losses = super().loss(preds_dicts, example, **kwargs)
        if self.sea is None:
            return losses
        at_gt = []
        for task_id, p in enumerate(preds_dicts):
            pred = p['anno_box'].transpose(1, 2)  # [B, K, 8], set by super
            tgt, sel, _ = get_corresponding_box(
                p['order'], example['ind'][task_id],
                example['mask'][task_id], example['cat'][task_id],
                example['anno_box'][task_id])
            zero = pred.new_zeros(*pred.shape[:2], 2)
            at_gt.append(dict(
                pred=torch.cat([pred, zero], -1),
                tgt=torch.cat([tgt, zero], -1), ind=p['order'],
                mask=sel.float(), score=p['scores'].detach(),
                log_b=p['sig'][:, 0], fb=p['fb'][:, 0]))
        self._metas = metas
        try:
            with torch.autocast('cuda', enabled=False):
                losses.update(self._sea_losses(at_gt))
        finally:
            self._metas = None
        return losses
```

- **完全复用** CenterPoint 头的 `_cell_to_m` 和 `_sea_losses`（第 50–51 行直接引用），保证两个检测器上的海面监督是同一份代码。已核对过：同样的输入下，三项损失的数值完全相等。
- 区别只在"从哪里读 GT 位置上的预测"：CenterFormer 不是稠密头，而是 500 个 query。训练时它会把 GT 位置强制放进这 500 个 query（`use_gt_training`），所以用 `get_corresponding_box` 找出"落在 GT 格子上的 query"作为观测。
- CenterFormer 的回归 code 只有 8 维 [偏移 2、z、log 尺寸 3、sin、cos]，在末尾补两个 0 当作 a2c，就和 CenterPoint 的 10 维 code 对齐了。
- heatmap 分辨率是 800×800（`out_size_factor = 4`），`_cell_to_m` 从 `train_cfg` 读取，自动适配。
- `post_processing` 重写了 NMS：原版只支持 3 个类别，这里改成按配置逐类做（5 类，IoU 0.2，和 CenterPoint 一致）。

---

## 7. 配置

**[gn_centerpoint-ea-sea_maritime-bench.py:1-40](configs/gn_centerpoint-ea-sea_maritime-bench.py#L1-L40)** · `whole file`

```python
"""CenterPoint (GroupNorm, 20 epochs) with evidence anchors and the
physical sea surface (SeaSurface): IMU tilt with a learned per-axis gain,
Kalman mean level, Gaussian-process wave field over hull footprints, and
per-detection waterline uncertainties, used as training supervision
(sea_at_test=False).
Ablations: gn_centerpoint-ea_ (no sea), gn_centerpoint-sea_ (centre anchor).
"""
_base_ = ['./gn_centerpoint-ea_maritime-bench.py']

model = dict(
    pts_bbox_head=dict(
        common_heads=dict(sig=(1, 2), fb=(1, 2)),
        sea_surface=dict(
            plane_scale=100.0,
            prior_c=-1.3,
            level_prior_var=1.0,
            wave_sigma=0.3,
            wave_length=25.0,
            tilt_gain=True,
            max_obs=50,
            hist_prob=0.5,
            hist_std_range=(0.05, 1.0),
            # train+val: the annotated level drifts ~4.9e-3 m^2 per 0.1 s and
            # IMU heave explains <1% of it (work_dirs/tmp/horizon/heave)
            kalman_q=0.005,
            heave_gain=0.0,
            max_gap=1.0),
        loss_final_height_weight=0.25,
        loss_sigma_weight=0.25,
        loss_level_weight=0.5,
        # training supervision only: at test time the detections keep the
        # head's own heights (fusing them with the sea surface at test time
        # raised ATE-z 0.71 -> 0.77 m and cost 1.9 test mAP3D at 80 ep;
        # the *-seatest_* configs evaluate that variant)
        sea_at_test=False))

# the mean level is streamed over each sequence: walk them in time order
stream = dict(sampler=dict(_delete_=True, type='SequentialChunkSampler'))
val_dataloader = stream
test_dataloader = stream
```

- `common_heads=dict(sig=(1, 2), fb=(1, 2))`：新增两个分支（会和基础配置里的 reg、height 等合并）；
- `sea_surface=dict(...)`：`SeaSurface` 的全部超参数，含义见 4.2 节；
- 三个损失权重 0.25 / 0.25 / 0.5；
- `sea_at_test=False`；
- `SequentialChunkSampler`：只有推理时用流式 Kalman 才需要；当前推理不用海面，这个 sampler 无害但不必要。

80 ep 版本（`gn_centerpoint-sea_maritime-bench-80e.py`）只改了训练轮数和学习率曲线。CenterFormer 版本（`gn_centerformer-sea_maritime-bench-80e.py`）用的是同一组海面超参数。

---

## 8. 推理开销

推理时不用海面，`SeaSurface` 和 sig、fb 两个分支都不需要计算。部署时可以把这两个分支（加上冗余的 a2c 分支，共约 0.57 M 参数）从模型里剪掉，得到和普通 CenterPoint **完全相同**的结构和速度。

（实测推理时间：CenterPoint 106 ms，CenterPoint + 海面监督 116–117 ms，推理时也用海面 118–121 ms。多出的约 10 ms 来自代码目前仍在计算 sig、fb、a2c 三个分支的卷积并多解码 4 个通道，剪掉即可消除。）

---

## 9. 实验结果

| 变体 | ep | test mAP3D | test mAPBEV | test ATE-z | >30 m 船的 ATE-z | val mAP3D |
|---|---|---|---|---|---|---|
| CenterPoint | 20 | 34.60 | 42.99 | 0.75 | 0.94 | 34.39 |
| **+ 海面监督** | 20 | **39.05** | 47.58 | 0.72 | 0.92 | 38.96 |
| + 海面监督，推理时也用海面 | 20 | 37.81 | 47.94 | 0.70 | 0.94 | 38.23 |
| CenterPoint | 80 | 48.14 | 56.07 | 0.72 | 0.90 | **46.06** |
| **+ 海面监督** | 80 | **49.24** | **58.18** | 0.71 | 0.92 | 42.53 |
| + 海面监督，推理时也用海面 | 80 | 47.38 | 57.83 | 0.77 | 1.03 | 42.23 |
| 最终模型（+ MariFusion） | 80+6 | 49.43 | 60.90 | 0.70 | 0.86 | 44.40 |
| 最终模型，推理时也用海面 | 80+6 | 49.39 | 60.90 | 0.74 | 0.91 | 44.02 |

怎么读：
- **作为训练监督**，test 上 20 ep 时 +4.5、80 ep 时 +1.1。
- **推理时也用海面**：20 ep 时还能接受（ATE-z 0.72 → 0.70），到了 80 ep 明显有害（ATE-z 0.71 → 0.77，>30 m 的船 0.92 → 1.03，mAP3D −1.9）。所以最终定为只在训练时使用。
- **val 和 test 不一致**：80 ep 时 val 上 −3.5，主要来自 ship（val 只有 4 艘）和 yacht（val 只有 3 艘）。按类别看，val 上 boat、buoy、sailboat 三类和 CenterPoint 基本持平。

---

## 10. 已知问题（需要你决定是否处理）

### 10.1 80 ep 模型学到的倾角增益符号反了

| | 纵摇增益 | 横摇增益 |
|---|---|---|
| **数据**：GT 底面拟合平面的斜率对 IMU 斜率回归（train，`up_causal`，10,731 帧） | **+0.34**（相关 0.15） | **+0.26**（相关 0.37） |
| 数据（val，3,606 帧） | +0.12（相关 0.07） | +0.24（相关 0.39） |
| 20 ep 模型学到的 | +0.62 | +0.33 |
| 80 ep 模型，第 55 ep | −0.44 | −0.24 |
| 80 ep 模型，第 80 ep | −0.49 | −0.27 |
| 最终模型（80+6） | −0.57 | −0.30 |
| CenterFormer（正在训练，第 11 ep） | +0.36 | +0.06 |

统计方法：每帧至少 4 个 GT 且 x、y 方向都跨度 40 m 以上，用 GT 底面拟合平面（去掉残差最大的 20% 后重拟合），再拿拟合斜率对 IMU 斜率做回归。

**结论**：
- **数据表明标注的高度确实部分跟随 IMU 倾角**：增益为正，约 0.3。横摇的相关性更强，纵摇较弱。
- 20 ep 的模型学到了合理的正增益；但训到 80 ep 时，增益**越过 0 变成了负数**。用 Adam 的步长粗略估算，纯随机噪声在 80 ep 内能造成的漂移大约只有 0.04，远小于实际的变化（约 1.1），所以这更像是有一个持续的梯度在推它，而不是随机漂移。
- 这**很可能就是"80 ep 时推理用海面变得有害"的原因**：倾角符号反了，离本船越远，高度误差越大，而 >30 m 的大船通常离得远，所以它们的 ATE-z 恶化最明显（0.92 → 1.03）。20 ep 时增益是正的，推理时用海面也没有害处。这是推断，没有直接验证。
- 当前方案只在训练时使用海面，所以这个问题**只会影响训练监督的质量**，不会直接影响推理结果。但它说明海面模型在训练后期有一部分"跑偏"了，**论文里不能写"学到了物理上合理的倾角增益"**。

可以考虑的修正（都需要重新训练）：
1. **把增益限制在 [0, 1.5]**，例如写成 1.5·sigmoid(·)，从参数化上禁止它变成负数；
2. **把增益固定为数据统计值**（纵摇 0.34，横摇 0.26），不再学习；
3. 先查清楚是什么在推它：例如在 80 ep 的权重上，分别对三个损失求增益的梯度，看是哪一项、在哪类场景下推的。

### 10.2 其他

- **平均海平面先验越来越松**：无历史时的先验标准差从 1 m 涨到 3.1 m，`prior_c` 从 −1.3 m 漂到 −2.9 m。在训练中有 50% 的帧带模拟历史，无历史的那一半帧里，模型实际上选择"几乎不相信先验"。这在物理上可以解释，但说明无历史先验没有起作用。
- **相关长度从 25 m 缩到 12 m**：和训练集"20 m 内相关 +0.52"大致吻合，偏短一些。
- **监督是间接的**：海面损失不直接作用于 `height` 分支（第 5.7 节）。论文里应该写成"物理约束的辅助监督"，不能写成"海面推理模块"。
- **a2c 分支是冗余参数**（约 0.19 M），保留是为了兼容已有权重。
- **配置里仍保留了推理时的流式组件**（`SequentialChunkSampler`、升沉查表），在当前推理不用海面时无害。

---

## 11. 常见问题

**Q1：还是端到端可学习的吗？**
是。`solve` 是闭式的线性代数运算（`torch.linalg.solve` 可以求导），三个损失的梯度都能流到 sig、fb 两个分支、海面参数和共享特征。只有观测的几何量（位置、尺寸、自身高度）被刻意 detach 了。

**Q2：为什么要 detach 观测的位置和高度？**
如果不 detach，海面损失会直接去改每个框的位置和高度，让框"迁就"海面：一个框的误差会通过共享的海面传染给其他框。旧版 SSR 出现过这个问题（框被压扁，高度只剩真值的 0.82 倍），所以这里只让海面损失去训练"可靠度"和"干舷"。

**Q3：为什么用 teacher forcing，而不用网络自己的检测结果？**
训练早期网络的检测很差，可能漏检，也可能偏得很远；用它们去估海面，监督信号会很嘈杂。在 GT 位置上读预测，可以保证每条真实的船都参与估计，并且位置是对的。推理时不需要海面，所以不存在训练与推理之间的不一致。

**Q4：方案 A（训练时模拟先验）具体是什么？**
真实推理时，上一帧的估计会作为这一帧的先验。训练时如果也要这样，就必须按时间顺序喂数据，而且要从头跑整个序列。方案 A 改为随机模拟：50% 的帧假设"有一个误差为 s 的历史估计"（s 随机），另外 50% 假设"没有历史"。这样训练照常随机打乱，网络也学会了在不同质量的先验下如何分配权重。

**Q5：为什么推理时不用海面，效果反而更好？**
80 ep 时，检测头自己的高度估计已经很准（ATE-z 0.71 m）。推理时再把框拉到海面上，相当于用一个"平均"的估计去覆盖一个已经很准的个体估计；再加上 10.1 节的倾角符号问题，结果就变差了。训练时的监督则不同：它只塑造特征，不覆盖输出。
