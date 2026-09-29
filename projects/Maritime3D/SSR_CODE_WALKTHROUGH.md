# SSR（共享海面推理）代码解析

版本：2026-09-26 · 对应代码：[maritime3d/wahead.py](maritime3d/wahead.py)（`WAHead` 类）、[maritime3d/sea_up.py](maritime3d/sea_up.py)、[maritime3d/samplers.py](maritime3d/samplers.py)

---

## 0. 一句话讲清楚 SSR

> **同一帧里的船都浮在同一片海面上。所以不让每个框各自猜高度，而是先让所有检测到的目标一起“投票”估计出这一帧的海面，再把每个框放到海面上。**

打个比方：一群人站在同一块地板上，你想知道每个人脚底的高度。与其对每个人单独测量（远处的人看不清，测得就不准），不如先用看得清的几个人把地板的高度定下来，再让所有人都站在这块地板上。

在这个比喻里：
- “地板”的**倾斜方向**由船上的 IMU（惯性测量单元）直接测出；
- “地板”的**高度**由检测到的目标一起估计；
- 相邻帧的海面几乎不变（0.1 秒内高度只差约 3 cm），所以还可以借用前几帧的估计（时序 SSR）。

---

## 1. 整体流程图

```
                    ┌──────────── 每帧一次 ────────────┐
 IMU 查表 (LoadSeaUp) ─→ 海面倾角 tilt=(a,b)          │
 前几帧的累积证据 ─────→ 历史 hist                     │
 可学习的全局先验 ─────→ prior=(a0,b0,c0)              │
                    └──────────────┬──────────────────┘
                                   ▼
点云 → 体素编码 → SECOND+FPN(GroupNorm) → BEV 特征 → heatmap 选 top-200 个 query
                                   │
              ┌────────────────────┴───── 对每一层 decoder 循环 ─────────────────────┐
              │ ① (可选) 把上一层的海面信息加到 query 特征上   (sea_context)            │
              │ ② decoder 层：query 去 BEV 特征里查信息                                   │
              │ ③ 预测头：每个 query 输出 中心、尺寸、朝向、原始高度 z_raw、干舷 fb、类别分  │
              │ ④ SSR：                                                                    │
              │     a. 每个 query 给出自己的“水线点”(x, y, 底面高 = z_raw − h/2) 和权重(得分²)│
              │     b. 加权最小二乘 + 先验 + 历史 → 解出这一帧的海面 π=(a,b,c)            │
              │     c. 最终高度 = 海面在(x,y)处的高度 + fb + h/2                           │
              └──────────────────────────────────────────────────────────────────────┘
                                   │
                    推理时：把这一帧的证据存起来，留给下一帧（时序）
                                   ▼
                     输出 3D 框（高度已经“贴”在海面上）
```

---

## 2. 关键概念（先把名词搞懂）

| 名词 | 含义 | 代码里的名字 |
|---|---|---|
| query | TransFusion 的“候选目标”，每帧 200 个，每个最后变成一个框 | `query_feat` |
| 原始高度 z_raw | 每个 query **自己**估计的框中心高度（和普通 TransFusion 一样） | `res['height']` → 改名为 `height_raw` |
| 底面 / 水线点 | 框底部的高度 = z_raw − h/2，也就是船贴着水面的位置 | `bottom` |
| 干舷 fb (freeboard) | 框底面相对海面的小偏移（残差），每个 query 单独预测 | `res['fb']` |
| 海面 π = (a, b, c) | 平面方程 z = a·x/100 + b·y/100 + c。a、b 是倾斜，c 是高度 | `plane` |
| 权重 ω | 每个 query 投票的分量 = 类别得分²，得分越高发言权越大 | `weight` |
| 先验 prior | 没有可靠证据时的默认海面，是可学习参数（c 初始为 −1.3 m） | `self.plane_prior` |
| λ | 先验的“分量”，相当于先验占 λ 个满分目标的票数 | `plane_prior_lambda` |

**坐标单位**：x、y 除以 `plane_scale=100`，是为了让 a、b、c 的数值大小差不多，解方程时数值更稳定。所以 a=1 表示每 100 m 升高 1 m（约 0.57°）。

---

## 3. 逐步讲解（按代码执行顺序）

### 第 1 步：准备这一帧的“海面线索”（每帧只做一次）

代码在 `forward_single` 的第 501–505 行：

```python
prior = self.plane_prior.float().expand(batch_size, 3)             # 全局先验 (a0,b0,c0)
tilt = self._imu_tilt(metas, prior) if self.imu_tilt else None      # IMU 给的倾角 (a,b)
solved = prior if tilt is None else prior[:, 2:]                    # 需要求解的参数
hist = self._history(metas, solved) if self.temporal else None      # 前几帧的证据
```

**① IMU 倾角**（[`_imu_tilt`](maritime3d/wahead.py#L361)）

LiDAR 是固定在船上的，船一晃，海面在 LiDAR 坐标系里就是斜的。IMU 能测出“重力朝上”的方向 `up`，把它换算成海面倾角：

```python
ab = torch.stack([-s * up[0] / up[2], -s * up[1] / up[2]])   # 倾角 (a, b)
rows.append(ab + self.tilt_offset.float())                   # + 可学习的安装偏差
```

- `up` 从哪里来：数据流程里的 [`LoadSeaUp`](maritime3d/sea_up.py)，按“序列 + 时间戳”到 `dataset/infos/detection/sea_up_lidar.npz` 里查表，默认用只看过去 5 s 的因果滤波结果（`up_causal`）。
- **关键细节**：训练时点云会做旋转、翻转、缩放增强，`up` 也必须跟着做同样的变换，否则就对不上了。`LoadSeaUp` 放在增强之后，并把 `lidar_aug_matrix` 作用到 `up` 上：
  ```python
  up = np.asarray(aug)[:3, :3] @ up     # sea_up.py
  ```
- 没有有效 IMU 数据的帧（约 0.1%），退回到先验的倾角。

**② 时序历史**（[`_history`](maritime3d/wahead.py#L322)）：见第 5 节。

### 第 2 步：decoder 每一层之后做 SSR

`forward_single` 的第 507–526 行是一个循环，每层 decoder 做完之后都调用一次 SSR：

```python
for i in range(self.num_decoder_layers):
    if ctx is not None and self.sea_context:            # (可选) 上一层的海面信息
        query_feat = query_feat + self.sea_embed(ctx)
    query_feat = self.decoder[i](...)                   # query 查 BEV 特征
    res_layer = self.prediction_heads[i](query_feat)    # 预测中心/尺寸/朝向/高度/fb/类别
    height, plane, evidence, ctx = self._ssr(res_layer, prior, hist, tilt)   # ← SSR
    res_layer['height_raw'] = res_layer['height']       # 保留 query 自己的高度估计
    res_layer['height'] = height                        # 换成“贴海面”的高度
    res_layer['plane'] = plane[..., None]
```

循环前面那段 query 初始化（heatmap 取 top-200）是原封不动照搬的 TransFusion。

### 第 3 步：SSR 本体（[`_ssr`](maritime3d/wahead.py#L419)）

**3a. 每个 query 报出自己的水线点和权重**

```python
weight = res['heatmap'].detach().sigmoid().max(1).values ** 2    # 权重 = 最高类别分²
cxy = ...anchor_to_center_cells(center, dim, rot).detach()       # 中心 (x, y)
if self.ssr_decouple:
    dim, h_raw = dim.detach(), h_raw.detach()                    # 截断梯度（见 4.1）
```

- 为什么用**得分的平方**：让高置信度目标主导投票，大量背景 query（得分约 0.001）几乎不起作用。
- `.detach()`：权重和位置只是“用来算海面的输入”，不希望海面的误差反过来去改动类别得分或位置。

**3b. 列方程**（[`_evidence`](maritime3d/wahead.py#L276)）

```python
bottom = h_raw[:, 0] - 0.5 * h                                   # 底面高度
if tilt is None:                                                 # 不用 IMU：解 (a,b,c)
    X = torch.stack([x / s, y / s, torch.ones_like(x)], dim=-1)
else:                                                            # 用 IMU：只解 c
    X = torch.ones_like(x)[..., None]
    bottom = bottom - (tilt[:, 0:1] * x + tilt[:, 1:2] * y) / s  # 先把倾斜“扣掉”
XtW = X.transpose(1, 2) * weight[:, None, :]
return XtW @ X, (XtW @ bottom[..., None])[..., 0]                # (A, b) = (XᵀWX, XᵀWz)
```

通俗地说：`A` 记录“有多少票、票分布在哪”，`b` 记录“票上写的高度加权后是多少”。两者一起就是这一帧的**证据**。

用 IMU 时，倾斜已经知道，每个水线点先扣掉倾斜造成的高度差，剩下的就只是“海面高度 c”，问题变成一个**加权平均**。

**3c. 解方程**（[`_solve_plane`](maritime3d/wahead.py#L297)）

```python
A = A + diag(λ)          ;  b = b + λ * prior        # 加上先验（λ 张“默认票”）
A, b = A + hist_A, b + hist_b                         # 加上前几帧的证据（时序）
return torch.linalg.solve(A, b)                       # 解出海面
```

这就是带先验的加权最小二乘（岭回归）。以只解 c 的情况为例，结果其实就是一个加权平均：

```
       Σ ω_i·(底面_i − 倾斜修正)  +  λ·c0  +  历史证据
c  =  ─────────────────────────────────────────────────
               Σ ω_i          +  λ   +  历史权重
```

- 目标多、置信度高时：Σω 大，c 主要由目标决定；
- 空帧或只有很弱的目标时：Σω≈0，c 退回先验 c0（有时序时退回前几帧的估计）。

**3d. 把框放到海面上**

```python
sea = self._plane_z(plane, x, y)              # 海面在每个 query (x,y) 处的高度
height = (sea + 0.5 * h)[:, None] + fb        # 最终中心高度 = 海面 + h/2 + 干舷
```

**3e. 海面上下文**（给下一层用，可选）

```python
ctx = cat([底面 − 海面高度（这个 query 和海面差多少）, a, b, c]).detach()    # 4 个数/query
```

下一层开始前，`sea_embed`（一个两层 1×1 卷积）把这 4 个数变成 128 维，加到 query 特征上，相当于告诉每个 query “你和大家的海面差了多少”。`sea_embed` 的最后一层**初始化为 0**，所以训练刚开始时它不起作用，不会打乱原模型。

---

## 4. 训练时怎么监督（[`loss`](maritime3d/wahead.py#L398)、[`loss_by_feat`](maritime3d/wahead.py#L588)、[`_plane_loss`](maritime3d/wahead.py#L619)）

总损失 = TransFusion 原有损失 + 两项新损失：

| 损失 | 监督对象 | 权重 | 目的 |
|---|---|---|---|
| 原有的 bbox L1（高度那一维） | **最终高度**（经过海面的那个） | 1 | 让“海面 + fb + h/2”贴近真值 |
| `loss_height_raw` | **原始高度** z_raw | 0.25 | 让每个 query 自己也学会估高度，它的投票才靠谱 |
| `loss_plane` | 海面 c（用 IMU 时）或 (a,b,c) | 0.5 | 让解出的海面贴近 GT |

**GT 海面从哪来（不需要额外标注）**：用 GT 框的底面算出来（[`_gt_sea`](maritime3d/wahead.py#L380)）。

```python
level = pts[:, 2] - (t[0] * pts[:, 0] + t[1] * pts[:, 1]) / s    # 扣掉 IMU 倾斜
return level.median()[None]                                       # 取中位数 = GT 海面高度 c
```

用 IMU 时只要**有 1 个框**就能算出 GT 的 c。不用 IMU 时要拟合整个平面，至少需要 3 个不共线的框。

### 4.1 梯度截断（`ssr_decouple=True`）为什么重要

如果不截断，一个框的高度误差会**通过共享的海面**，反向影响到**其他所有框**的高度和尺寸。实验中，这让模型学会了“把框压扁”：底面偏高 0.75 m，框高只有真值的 0.82。

截断之后：
- “经过海面的最终高度”这一项损失，只训练**干舷 fb** 和**全局先验**；
- 框的尺寸 h 和原始高度 z_raw 仍然由它们自己的损失训练，和普通 TransFusion 一模一样。

修好后，底面偏差回到 +0.03 m。

---

## 5. 时序 SSR：借用前几帧（`temporal=True`）

**动机**：大多数帧里高置信度目标很少（每帧权重之和的中位数只有约 0.05–0.4），单帧证据很弱。但海面变化很慢：相邻帧高度只差约 3 cm。所以可以把前几帧的证据**累加**起来用。

**推理时**（[`_history`](maritime3d/wahead.py#L322) + [`_update_stream`](maritime3d/wahead.py#L353)）：

```python
# 取历史：同一序列、间隔 0 < dt ≤ 1 s 才用，按时间衰减
decay = self.temporal_forget ** (dt / 0.1)      # 每过 0.1 s 乘 0.9
H, h = decay * 上一帧存的A, decay * 上一帧存的b

# 这一帧算完后，把 (本帧证据 + 衰减后的历史) 存起来给下一帧
self._stream[seq] = (ts, A + H, b + h)
```

这就是带遗忘因子的“递推最小二乘”：越近的帧权重越大，大约记住最近 1 秒。序列号和时间戳是从点云文件名 `points4/<seq>/<ns>.bin` 里解析出来的（[`_stream_key`](maritime3d/wahead.py#L313)）。

**配套的评测采样器**（[`SequentialChunkSampler`](maritime3d/samplers.py)）：默认的采样器会把帧轮流分给 8 张卡（卡 0 拿第 0、8、16… 帧），这样每张卡看到的帧就不连续了。新采样器让每张卡拿**一整段按时间排序的连续帧**。它还处理了 mmengine 汇总结果时的补齐问题，保证每一帧正好被评测一次。

**训练时**：训练的帧是打乱的，没有“前一帧”。所以以 50% 的概率，用“GT 海面 + 随机噪声”当作一个假的历史，随机给它 0–10 的权重，让模型习惯“有时有历史、有时没有”。

---

## 6. 配置开关一览

主方法配置：[configs/gn_wahead-ssr-imu-temporal_transfusion_lidar_maritime-bench.py](configs/gn_wahead-ssr-imu-temporal_transfusion_lidar_maritime-bench.py)

继承关系：
```
transfusion_lidar_maritime-bench.py            (v2 基线)
 └ wahead_transfusion_lidar_maritime-bench.py   (换成 WAHead，加 fb 头，SSR 超参数都在这里)
    └ wahead-ssr-only_...                      (kappa=0：关掉已证伪的 VWA)
       └ wahead-ssr-temporal_...               (temporal=True, ssr_decouple=True, 评测用顺序采样器)
          └ wahead-ssr-imu-temporal_...        (imu_tilt=True, 数据流程加 LoadSeaUp)
             └ gn_wahead-ssr-imu-temporal_...  (骨干网络 BN → GroupNorm)
```

| 参数 | 默认 / 主方法取值 | 作用 |
|---|---|---|
| `ssr` | True | 总开关；False 时就是原版 TransFusion |
| `ssr_decouple` | True | 梯度截断（第 4.1 节） |
| `imu_tilt` | True | 倾角用 IMU，只解 c |
| `temporal` | True | 时序累积 |
| `temporal_forget` / `temporal_max_gap` | 0.9 / 1.0 s | 每 0.1 s 的衰减系数 / 超过多久重置 |
| `sea_context` | False（2 层 decoder 的完整版为 True） | 把海面信息加回下一层 |
| `plane_prior_lambda` | (1, 1, 1) | 先验的票数 |
| `plane_prior_c` | −1.3 m | 海面高度先验的初始值 |
| `loss_plane_weight` / `loss_height_raw_weight` | 0.5 / 0.25 | 两项新损失的权重 |
| `LoadSeaUp.up_key` | `up_causal` | 因果（在线）；`up` 是非因果的离线上限，评测时可以切换 |

---

## 7. 踩过的坑（读代码时会看到相关注释）

| 问题 | 现象 | 现在的处理 |
|---|---|---|
| 先验原来来自“全局池化特征 + MLP” | 训练和推理时 BN 统计不同，先验从 −2.5 m 漂到 −10.7 m，所有框被拉低 3.8 m | 改为可学习常数 `plane_prior` |
| 梯度耦合 | 框被压扁，底面偏高 0.75 m | `ssr_decouple` |
| 时序放大偏差 | 证据本身有偏时，累积越多越偏 | 先修耦合，再开时序 |
| 骨干网络 BN 在训练和推理时不一致 | 推理时框缩小、朝向误差到 40°+，而且因 checkpoint 而异 | SECOND/FPN 改为 GroupNorm（`gn_*` 配置） |
| 用 GT 框拟合海面倾角 | 倾角大部分是标注噪声（约 0.7°） | 倾角改用 IMU，GT 只用来算 c |

---

## 8. 想动手验证时看这些

- 单元测试（可以当使用示例）：`work_dirs/tmp/test_imu_ssr.py`（IMU 模式）、`work_dirs/tmp/test_temporal_ssr.py`（时序和采样器）
- 逐维误差诊断：`work_dirs/tmp/diag_err.py <config> <ckpt>`（看 dz、尺寸比例、yaw 误差）
- 时序开关对比：`work_dirs/tmp/eval_temporal_ssr.py <ckpt>`
- 历次实验记录：`progress/` 目录，按时间戳排列
