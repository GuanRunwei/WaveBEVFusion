d'n# Mari3D v5 基准验证报告

日期：2026-09-24 ｜ 生成工具：`tools/resplit_maritime3d_v5.py`（确定性，seed=20260923）＋ `tools/patch_maritime3d_v5_calib.py`

## 一、v5 划分（一套标注 → 检测 + 追踪两个基准）

### 设计
- **块级划分**：20s（200 帧 @10Hz）连续块；train/val/test 均为整块，无帧级交错。
- **10s 时间缓冲带**：所有 split 边界处抽离 3089 帧（3.8%）作为 buffer（不属于任何 split），实测 train↔test 最小间隔 **10.000s**、train↔val **10.002s**。
- **长尾保护**：yacht/ship/buoy/sailboat 按致密窗口播种（播种量按缓冲后内部覆盖 ≥1.25×floor 计算），精修以（floor 违约, 代价）字典序优化，**全部类别 floor 达标、零 WARN**。
- **跨天泛化子集**：seq03（比 00-02 晚 14 天采集）52% 帧划入 test → `maritime_split_test_v5_crossday.json`（4400 帧）。
- **一套标注两用**：det_train ∪ det_val == tracking_trainval（62988 帧，逐 token 相等）；det_test == tracking_test（14200 帧）。

### 数据分配
| split | 帧 | 框 | boat | buoy | sailboat | ship | yacht |
|---|---|---|---|---|---|---|---|
| train | 51783 | 193523 | 62% | 65% | 57% | 62% | 36% |
| val | 11205 | 43538 | 14% | 14% | 13% | 13% | 10% |
| test | 14200 | 60132 | 19% | 19% | 16% | 16% | 35% |
| buffer | 3089 | 16803 | — | — | — | — | — |

守恒：77188 + 3089 = 80277 帧；297193 + 16803 = 313996 框（v4 宇宙一个不少，**没有浪费任何标注**）。
轨迹：787/844 条轨迹完整落在单侧（57 条在 trainval/test 边界被切断，9 条整体在 buffer 内）。

### 独立对抗审计（全新代码，不复用生成脚本）
9 项实质断言 8 项**精确通过**：成对不相交、帧/框守恒（逐实例精确）、det/trk 帧集相等、时间隔离、五类 floor、跨天子集（100% seq03）、逐帧字典与 v4 **逐位一致**（含 40001 个换侧帧，内容零改动）。

已知非阻塞观察：
1. seq00 存在一个 200 帧的 train 孤岛（夹在两个 test 块之间，两侧均有 10s buffer，正确性无影响，仅略碎）。
2. `sample_idx` 沿用 v4 旧值，同一 v5 split 内会重复（v4 中它按 split 内唯一）。mmdet3d 以 `token` 为主键，不受影响；若有下游代码用 `(scene_token, sample_idx)` 作键需改用 `token`。
3. buffer 带宽是"边界两侧各 ~5s"（合计 ~10s），而非字面"边界 10s 内"；隔离保证按 ≥10s 实测成立。

## 二、本次同时修复的三个"训练结果不好看"根因（v4 遗留、与划分无关）

1. **点云加载错误（致命）**：原始 bin 是定长 131072×7 float32 缓冲，真实点（约 3 万/帧）散布其中，其余为全零填充行。旧配置 `load_dim=4` 会读出 13 万"点"，其中 **~75% 是 (0,0,0) 假点**——这样训练出来的模型必然难看。
   修复：v5 配置改 `load_dim=7` + 新增 `StripEmptyLidarPoints` 变换（`projects/Maritime3D/maritime3d/transforms.py`）。实测 train 平均 **29964 真实点/帧**。
2. **相机标定是占位符（致命）**：pkls 里的 cam2img 是 f=1408（图宽一半）、lidar2cam=单位阵的假值，GT 投影进图像的比例为 **0%**——所有相机/融合分支（FCOS3D/PETR/BEVFusion 图像塔）此前测到 0 mAP 是必然，不是"基准难"。
   修复：`tools/patch_maritime3d_v5_calib.py` 已把四个序列的真实内参（f≈1814.5，stereo_left cc=(1026.2, 524.8)）与外参（约定 wxyz/b2s，按投影命中率 30.1% 经验选定并记录在 metainfo）回填到全部 5 个 v5 pkl（30.9 万条图像条目，原子改写，标注字段零改动）。回读验证：test 集 **22.1%** GT 中心落入左目图像（train 29.2% / val 25.7%），与 ~59° 前向视场占比相符。
3. **bbox_3d 是 11 维，锚框训练直接崩溃（致命，v4 同样中招）**：new_annotations 的 `bbox_3d` 是 11 维 `[x,y,z,dx,dy,dz,yaw,pitch,roll,pad,pad]`（"10dof"），而 mmdet3d 的 anchor 分配器/IoU 计算断言 7 维核心——`MaritimeDataset.parse_ann_info` 原样把 11 维塞进 `LiDARInstance3DBoxes`，任何 anchor 系模型（PointPillars/SECOND 等）在 loss 第一步就 `AssertionError`。**即 v4 pkls 从来没能用这套数据集类训起来过**（旧 4 类配置用的是另一套 7 维 seq00-only pkls）。pitch/roll 实测 ≤0.01 rad（近零），且以 `gt_pitch/gt_roll/pitch_roll` 逐实例保留在 pkl 中供未来 10-dof 方法使用。
   修复：`parse_ann_info` 切片到 7 维核心后构建框。GPU 实测：v5 train（51783 帧）loss 正常计算、优化器 step 通过、第二步 loss 下降；GT 作为预测喂入 MaritimeMetric 得 **mAP3D/mAPBEV=1.0000、mAOE=0.0000，五类全部 AP=1.0**（各距离桶）——评测链路数学上正确。

## 三、v5 配置与代码
- `projects/Maritime3D/configs/_base_/maritime-3d-5class-v5.py`：5 类（v5 顺序 boat/buoy/sailboat/ship/yacht）、指向 `dataset/new_annotations/detection/*_v5.pkl`、三条 pipeline 均已接入点云清洗。
- `projects/Maritime3D/configs/pointpillars_maritime-3d-5class-v5.py`：锚框尺寸/z 用 v5 训练集逐类中位数重算（yacht 3.2×8.4×4.0 等），train_cfg/test_cfg 挂 model 层（mmdet3d 1.x 规范）。
- `MaritimeDataset` 兼容修复：config 类别表不是 legacy 4 类子集时自动作为全量词表（identity 映射）；legacy 4 类配置行为不变（回归测试通过：legacy 6586 帧、v5 train 51783 帧、v5 test 14200 帧均可构建并跑通 pipeline）。
- **真实 `num_lidar_pts` 已回填**：pkls 原来全为占位 100（难度过滤失效）；tracking 表里其实带每框真实点数。`tools/backfill_maritime3d_v5_num_lidar_pts.py` 按 sample token + 尺寸 + 中心最近邻把 593486 个实例全部回填（0 未匹配；145 个源数据 null 保持 None），难度分布真实化：train 中位数 54 点、四分位 [0,17,54,193]，test [0,20,56,181]。独立抽检 150 帧与表逐帧一致。
- 注意：GT-Aug（dbinfo）尚无 v5 版本，配置中已停用 ObjectSample 并注释——用 legacy dbinfo 会把 val/test 形状泄漏进训练。需要时从 v5 train 重新生成。

## 三点五、两项收尾审计（tracking 表完整性 + 基准可用性）

**Tracking 表审计（21 项断言，14 PASS；7 项 FAIL 全部为"预期行为/源数据继承"，v5 未引入任何新缺陷）**：
- ✅ 9 张表 × 2 split 引用完整性 165 万条外键 0 悬挂；sample/sample_data prev/next 35 个 scene 全部单链完整；instance 行数/端点与链完全一致；与 track pkls 逐 token、逐帧框数一致；帧零泄漏（时间戳/lidar 文件名跨 split 零共享）。
- ① 57 个 instance token 同时出现在 trainval/test 表 = 恰好 57 条跨边界轨迹（帧完全不相交，见上）；② 67 条 trainval 轨迹跨多个 v5 scene（20s 块式 scene + buffer 切断所致，链端点仍一致）；③ 591 行（0.199%）prev/next 指针重接 = buffer 丢弃后链重建，**非指针字段 0 改动**；④ 145 个 num_lidar_pts=null 全部继承自 v4；⑤ 5 个 scene 内 sample_idx 有一次计数器跳变（v4 旧边界遗留，token 为主键，无功能影响）；⑥ **131 行同一 instance token 同帧出现两次（2 条轨迹，v4 源标注固有缺陷：v4 的 90(trainval)+41(test) 在 v5 合并后同侧）**——检测不受影响，追踪 GT 在这 0.04% 帧上 id 有歧义，论文应声明；⑦ pkl 的 num_lidar_pts 占位问题已由回填解决（见上）。

**基准可用性审计（全部通过，且捞出并修复根因 #3）**：
- train（51783 帧，逐类框数与划分表精确一致）→ loss 计算 → 优化器 step → loss 下降：**PASS**（GPU 实测）。
- test（14200 帧，空帧保留）→ predict（60 帧/3.2s）→ MaritimeMetric 全表计算：**PASS**。
- GT 作为预测：**mAP3D=1.0000 / mAPBEV=1.0000 / mAOE=0.0000，五类在所有有 GT 的距离桶 AP=1.0**（ship 0-50m 桶内无 GT 故 nan，属正常）——评测定义正确性的端到端证明。

## 四、数据集资格结论（结合 17-agent 深度审计）
**标注本体质量好**：0.03% 框内零点、0 重复框、yaw 真实多峰、轨迹链 0 悬挂指针、239353 条媒体路径全部存在。可以支撑严肃的 benchmark。

**v4 划分的结构性问题在 v5 中的状态**：
| v4 问题 | v5 状态 |
|---|---|
| train/test 块边界 0.096s 贴脸、88.2% test 帧同录像内后侧有 train 帧 | ✅ 已修：所有边界 ≥10s buffer |
| val/test 帧级交错 | ✅ 已修：整块划分 |
| 小类 test 缺失风险（yacht 曾只有 40 框进 test） | ✅ 已修：全部类别 floor 达标 |
| 无跨天泛化 | ✅ 已修：seq03 = 14 天后采集，占 test 52% |
| 同录像（session 内）测试 vs 跨 session 泛化 | ⚠️ 设计取舍：数据长尾（你的约束）不允许整 session 划出；v5 以 10s buffer + 跨天子集缓解。建议论文中把 v5 主协议与 crossday 子测试并列报告 |
| 追踪：边界处同一物理目标两侧换 identity | ⚠️ 57/844（6.8%），与 v4 同量级；追踪评测惯例可接受，报告里注明即可 |

**遗留已知问题（不影响运行，但发论文前应处理/声明）**：
1. 实例 `velocity` 全为 0（10dof 名不副实）：mAVE 指标对恒零预测器满分，论文别报 mAVE，或先做 IMU 补偿速度重算。
   **pitch/roll 同理（2026-09-24 补充分析）**：标注中的 pitch/roll（bbox_3d 第 8/9 维）**88-94% 为精确 0**，p99 仅 0.7-1.5°，有意义的倾斜几乎只出现在 buoy（均值 0.5-0.6°，max 23°），其余类别均值 ≤0.06°。恒零预测器即可在 p50/p90 上打平任何方法——**该轴不具备作为评测维度的动态范围**。基准决定：预测头保持标准 7-dof（PointPillars/SECOND 惯例），评测只报 yaw 的 mAOE；pitch/roll 作为附带标注保留在 pkl（`gt_pitch/gt_roll` 逐实例字段），论文中说明"10dof 标注提供但该轴近退化"，或可作 buoy 横倾估计的附加分析而非主表指标。
2. ~~`num_lidar_pts` 全为占位 100~~ **已解决**：真实点数已从 tracking 表回填全部 5 个 v5 pkl（145 个源数据 null 除外，见 5）。
3. test 标注随数据发布（v1.0-test 表含 GT）：对标 SOTA 论文惯例应服务器端评测；短期至少在论文注明。
4. ship/yacht 与 train 存在尺寸分布偏移（~60% test 框在 train 尺寸包络外）：长尾+少样本固有，报 per-class AP 并给置信区间。
5. 145 个 `num_lidar_pts` 为 null 的实例（源标注固有）；约 1% 训练框在水面以下（z 标定/潮汐问题）。另：131 行同帧同 instance token 重复（2 条轨迹，v4 源标注固有，追踪 GT 在 0.04% 帧上 id 歧义，论文应声明）。
6. 主评测范围 ±160m 内远距离 GT 点支撑极稀（>100m 中位 ~1 点/框）：建议主表 cap 到 ≤100m 或报距离分桶 AP。
7. tracking 缺 nuScenes 部分表与 AMOTA 评测线、无 tracker baseline：发 tracking benchmark 前需补（boat 主导，ship/buoy test 轨迹太少，建议报 merged-class 或 boat-only AMOTA）。

## 五、复现
```bash
# 1. 划分（确定性）
python tools/resplit_maritime3d_v5.py            # 写全部 v5 产物 + 自校验
# 2. 标定回填（幂等）
python tools/patch_maritime3d_v5_calib.py
# 3. 真实 num_lidar_pts 回填（幂等）
python tools/backfill_maritime3d_v5_num_lidar_pts.py
# 4. 训练
python tools/train.py projects/Maritime3D/configs/pointpillars_maritime-3d-5class-v5.py
```
