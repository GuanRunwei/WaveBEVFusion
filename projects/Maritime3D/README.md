# maritime3d — a 3D object detection benchmark for surface vessels

Sequence 00 of the maritime3d recordings, converted to the OpenMMLab v2 info
format with five reference baselines: three LiDAR-only, one LiDAR + stereo
camera fusion, one camera-only.

Everything below is reproducible from this repository:

```bash
python tools/make_splits.py                 # temporal-block train/val/test split
python tools/convert_maritime.py            # KITTI labels + calib -> info pkls
python tools/create_gt_database.py          # ObjectSample database
python tools/extract_ego_attitude.py        # IMU -> per-frame hull attitude
python tools/analyze_attitude_sensitivity.py  # what ignoring pitch/roll costs
bash   tools/run_maritime_benchmark.sh      # train + test all five baselines
python tools/collect_maritime_results.py    # results table
```

## 1. Sensors

| | |
|---|---|
| LiDAR | 1 forward-mounted unit, ~16k points/frame (median), returns out to ~150 m |
| Cameras | 2 (`CAM_LEFT`, `CAM_RIGHT`), 2048x1080, f ~ 1813 px, 58.9 deg HFOV, 0.84 m baseline |
| Camera pose | optical axes at -5.4 deg azimuth, +2.7 deg elevation, 1.36 m above the LiDAR origin |
| IMU | 96.4 Hz, co-located with the LiDAR; gyro + accelerometer, no magnetometer |
| Annotation | KITTI-style 7-DoF boxes (x, y, z, dx, dy, dz, yaw), full 360 deg, out to 178 m |

The rig is the single most important thing to understand about this benchmark:
**the LiDAR and the annotations cover the full circle, the cameras cover about
59 deg ahead.** Every design decision below follows from that asymmetry.

All frames are converted to a common LiDAR frame with +x along the vessel
heading, +y to port, +z up. Evaluation volume is
`[-160, -160, -8] .. [160, 160, 24]` m.

## 2. Classes

| class | all | train | val | test |
|---|---|---|---|---|
| boat | 83676 | 51447 | 7083 | 14073 |
| ship | 751 | 391 | 130 | 130 |
| sailboat | 5593 | 3357 | 596 | 875 |
| buoy | 1746 | 1082 | 164 | 276 |
| **total** | **91766** | **56277** | **7973** | **15354** |

All four classes are kept and reported separately. The imbalance is extreme --
`boat` outnumbers `ship` by 110:1 -- so a single averaged mAP hides most of what
a method does. Read the per-class columns, not the mean.

### Object counts are not sample sizes

The table above counts annotated boxes, and for a 10 Hz recording that badly
overstates how much independent evidence there is. The 130 `ship` boxes in the
test split are **one vessel**, tracked across 130 consecutive frames, drifting
0.10 m per frame. Counting *encounters* instead -- a maximal run of frames
containing the class, split wherever more than 15 frames (1.5 s) of the original
sequence pass without it -- gives the real support:

| class | train | val | test |
|---|---|---|---|
| boat | 20 | 8 | 12 |
| ship | 3 | 1 | **1** |
| sailboat | 4 | 1 | **2** |
| buoy | 10 | 2 | **4** |

Three of the four classes -- 75% of an unweighted mAP -- rest on **7 unique
object encounters** in the test split. `ship AP` is not a class-level
measurement at all; it is the answer to "did the detector find this one
vessel", reported to two decimal places.

This is the single most important caveat in the benchmark, and it is why
`tools/bootstrap_maritime_ci.py` resamples encounters rather than frames: a
frame-level bootstrap would treat 130 views of one ship as 130 independent
draws and report an interval far tighter than the truth. See §6.

## 3. Splits

**7:1:2 by temporal block, not by frame.**

The requested 7:1:2 ratio is kept, but a uniform random frame-level split would
have made the benchmark meaningless: consecutive frames in a 10 Hz recording are
near-duplicates, so a random split puts the same vessel, at nearly the same
range and pose, in both train and test. Reported numbers would then measure
memorisation rather than generalisation -- the failure mode that made
frame-level splits unacceptable on KITTI and nuScenes too.

Instead:

* the 22184 annotated frames are cut into 148 contiguous blocks of 150 frames
  (~15 s each);
* blocks are assigned 103 / 15 / 30 to train / val / test, which is 7:1:2 at
  block level;
* a 20-frame buffer at each block boundary is **dropped entirely** (2960 frames)
  so that no test frame is within 2 s of a training frame.

| split | frames | non-empty | objects |
|---|---|---|---|
| train | 13390 | 6586 | 56277 |
| val | 1950 | 1041 | 7973 |
| test | 3884 | 1875 | 15354 |

Block assignments are recorded in `data/maritime/ImageSets/split_stats.json` and
the frame lists in `ImageSets/*.txt`, so the split is fixed and citable rather
than regenerated per run.

### Empty frames

Half the sequence contains no annotated object (open water). These are
**dropped from training** -- they contribute no positive supervision and would
halve the effective epoch -- but **kept in val and test**, where a detector that
hallucinates vessels on empty water must be penalised for it. Hence the two
training lists: `maritime_infos_train.pkl` (all frames) and
`maritime_infos_train_nonempty.pkl` (the one the baselines actually use).

### Honest limits of this split

This is one sequence, one vessel, one geography, one weather condition. It is a
usable benchmark for comparing detectors on this data; it is not yet grounds for
claiming a method generalises to maritime perception in general. Making this
authoritative needs additional sequences held out as a *cross-sequence* test
set, which is the natural next step once more sequences are annotated. Until
then, the test split measures generalisation across time within a sequence, and
that is all it should be quoted as measuring.

## 4. Metric

`MaritimeMetric` (`projects/Maritime3D/maritime3d/maritime_metric.py`) reports
KITTI-style **AP40** at both 3D and BEV IoU.

| class | IoU threshold |
|---|---|
| boat, ship, sailboat | 0.50 |
| buoy | 0.25 |

Buoys get a looser threshold for the same reason KITTI gives pedestrians and
cyclists one: they are ~2.5 m across, so a 0.5 m localisation error already
costs more IoU than it does on a 77 m ship.

Results are also broken down by range (0-50 / 50-100 / 100-160 m). Median object
range is 77 m -- far beyond the 0-70 m KITTI regime -- so an aggregate AP is
dominated by the mid-range band and says little about the long-range behaviour
that matters at sea.

**BEV IoU uses `box_iou_rotated` on the `.bev` representation**, matching
`BaseInstance3DBoxes.overlaps`. This is worth stating because the obvious
alternative, `mmcv.ops.boxes_iou_bev`, silently expects `[x1, y1, x2, y2, ry]`
corner format; feeding it centre-format boxes produces AP@BEV *below* AP@3D,
which is geometrically impossible and was caught only by asserting
`bev >= 3d` on random boxes.

### Centre-distance AP

The same AP40 is also reported with detections matched to ground truth by **BEV
centre distance** at 2 / 5 / 10 / 20 m, alongside the IoU numbers and never in
place of them.

IoU 0.5 on a 6 m boat demands roughly 2 m of centre accuracy. The stereo pair
has a 0.84 m baseline, worth 9-15 px of disparity at 100-160 m, and PETR reads
the two cameras as independent monocular views rather than matching them, so its
depth error out there is ~19 m. Every AP@IoU it produces is therefore exactly
0.00 -- and a column of zeros cannot distinguish a model that learned nothing
from one that is consistently 10 m off in range while getting bearing right to
9 deg. nuScenes reports centre distance for this reason; the thresholds here are
its 0.5/1/2/4 m ladder scaled to a domain whose median object sits at 77 m
rather than 20 m.

Read it as "is there a vessel about there", not "is this box right": it ignores
extent and heading entirely, so it is strictly the weaker claim. The IoU columns
remain the primary result for LiDAR and fusion methods.

### Euler angles: what is scored, and what cannot be

> Full analysis in **[EULER_ANGLES.md](EULER_ANGLES.md)** -- the observer /
> target split, why pitch costs more IoU than roll on every class except the
> sailboat, what 5 deg of unmodelled target pitch measurably costs, and what a
> 9-DoF re-annotation should prioritise. Summary below.

The sea is not a road. A driving benchmark can assume the rig stays level, so a
7-DoF box in the sensor frame is also a box in the world frame. Here the hull
moves in all three angles, and the IMU at
`dataset/00/lidar_front/imu.txt` (213781 rows, 96.4 Hz) says by how much:

| | RMS | dominant period | extremes |
|---|---|---|---|
| roll | 0.70 deg | 2.25 s (wind waves) | -8.3 .. +4.9 deg |
| pitch | 1.35 deg | 19.0 s (swell) | -3.2 .. +3.9 deg |

That is not a second-order correction at maritime ranges. One degree of tilt
displaces a target 2.6 m at 150 m, and IoU 0.5 on a 6 m boat allows about 2 m of
centre error in total. Attitude is the dominant geometric term out there.

**Yaw is scored. Pitch and roll cannot be, and this is a property of the data,
not a choice.** The KITTI-format labels carry 15 fields plus a constant `-1`;
field 15 is `rot_y` and there is nothing after it. No target's pitch or roll is
annotated anywhere in the sequence, and every baseline here -- SECOND,
PointPillars, TransFusion-L, BEVFusion, PETR -- regresses a single heading angle
by construction. There is nothing to compare against and nothing to compare.
Closing this needs 9-DoF re-annotation and 9-DoF detection heads; it is listed
in §7 rather than papered over with a metric that would silently score zero.

What the data *does* support is two things:

* **Orientation error (AOE)** on the angle that is annotated. Mean absolute yaw
  error over detections matched within 10 m of BEV centre distance -- matched on
  distance rather than IoU, because matching on IoU would measure heading only
  where the box is already good. Reported in two columns: `AOE` wraps to
  `[0, 180]` and so counts a bow/stern swap as a 180 deg error, `AOE180` folds
  to `[0, 90]` and does not. A gap between them means the heading *axis* is
  right and the direction along it is not, which is a different defect from a
  box that is simply skewed.
* **AP conditioned on sea state.** `tools/extract_ego_attitude.py` reconstructs
  per-frame hull attitude and writes `data/maritime/ego_attitude.npz`; the
  metric then bands the test frames by how hard the hull is rocking and reports
  AP in each band. Per-object pitch and roll are unannotated, but the
  *platform's* are measured, and they are what make a flat-water prior wrong. A
  method that holds its AP from the calm band to the rough one is robust to
  wave motion on this recording; one that does not has been leaning on a
  level-sensor assumption.

Attitude is reconstructed with a complementary filter rather than from either
sensor alone, because neither alone is usable: the accelerometer measures
gravity *plus* hull acceleration, so taken raw its roll swings 87 deg peak to
peak, which is slamming, not attitude; and the gyro integrates cleanly but
carries a -1.79 deg/s bias that would accumulate to thousands of degrees over
the 37-minute record. The two triads are consistent with each other -- accel
d(roll)/dt correlates +0.79 with gyro_x at slope +0.99, pitch with gyro_y at
+0.69 / +1.08, both cross terms below 0.19 -- so the fusion is well posed.

#### Sea state is banded on rate, not on tilt

The obvious sea-state variable is how far the hull is leaning. It is the wrong
one, and measurably so. An accelerometer cannot separate a roll from the
centripetal acceleration of a sustained turn, so a two-minute turn reads as two
minutes of heel; on this record the tilt estimate correlates +0.30 with
sustained yaw rate, and frames in the top tilt tercile are turning at 1.28 deg/s
against 0.63 deg/s in the bottom one.

That matters because turning is not independent of what is in the scene. The
first version of this breakdown banded on tilt and put **11233 boat boxes at a
median 84 m in the calm band against 194 boxes at a median 14 m in the rough
one**. Every method showed a large AP drop across those bands, and none of it
was attributable to wave motion: the two bands hold different vessels at
different ranges, not the same vessels in different conditions. That result was
discarded.

The banding variable is instead the **RMS of the roll+pitch rate over a 4 s
window**, in deg/s. A steady turn is near-DC in rate and a static trim offset is
exactly DC, so both drop out and what survives is the oscillation waves actually
cause. Band edges are the terciles of the full annotated record, 0.77 and
1.48 deg/s, fixed as constants so they stay comparable as sequences are added.
The resulting test bands are matched on everything that would otherwise explain
a gap:

| | calm | moderate | rough |
|---|---|---|---|
| test frames | 1174 | 1623 | 1087 |
| temporal blocks | 21 | 26 | 23 |
| boat boxes | 3996 | 7844 | 2233 |
| median boat range | 82.4 m | 80.6 m | 78.3 m |
| boats beyond 100 m | 33% | 27% | 37% |

A difference between these bands is therefore attributable to the motion. The
class imbalance is not fixed by this -- `ship` still has no encounter in the
rough band and `buoy` none in the calm one -- so only the `boat` column is
readable, and the `mAP` columns in RESULTS.md must be ignored.

One caveat on interpreting the bands: the point cloud is not raw body-frame
data. Ground-truth `z` scatter grows with range (0.88 / 1.05 / 1.33 m over the
three range bands) but well below the 0.59 / 1.77 / 3.06 m a fully body-fixed
cloud would produce, and its correlation with the IMU-predicted tilt offset
decays from +0.47 at 0-50 m to +0.06 at 100-160 m. Some levelling has been
applied upstream. The sea-state breakdown therefore measures residual
sensitivity to hull motion, not the full uncorrected effect.

### Camera-sector protocol

The camera-only baseline is additionally scored inside the sector its cameras
actually see, `[-35, +24]` deg, under the `MaritimeFOV/` prefix. Its full-circle
mAP is reported too so it sits in the same table as everything else, but that
number is bounded by the rig -- roughly 78% of ground-truth objects are outside
any camera frustum and no camera-only method can recover them. Quote the
full-circle number as an upper bound on a forward-facing camera-only system, and
the sector number as the measurement of the method.

## 5. Baselines

| config | modality | notes |
|---|---|---|
| `pointpillars_maritime-3d-4class.py` | LiDAR | pillars, anchor head |
| `second_maritime-3d-4class.py` | LiDAR | 3D sparse conv, anchor head |
| `transfusion_lidar_maritime-3d-4class.py` | LiDAR | same backbone as SECOND, transformer head |
| `bevfusion_lidar-cam_maritime-3d-4class.py` | LiDAR + stereo | Swin-T + DepthLSS, ConvFuser |
| `petr_maritime-3d-4class.py` | stereo camera | ResNet-50 + CPFPN, 3D position-aware queries |

Shared settings, so that differences in the table are attributable:

* **One BEV grid for every LiDAR baseline**: voxel 0.3125 m over +-160 m gives
  1024x1024, downsampled 8x to a 128x128 BEV map. 0.3125 rather than KITTI's
  0.16 because the detected area is ~19x larger (102400 m2 vs KITTI's 5485);
  1024 rather than 1000 because `SECONDFPN`'s stride-2 branch needs an even
  downsampled size (1000/8 = 125 is odd, and the upsample comes back as 126).
* **TransFusion-L and BEVFusion share a byte-identical LiDAR branch**, so their
  difference isolates the camera contribution.
* **40 epochs**, one pass per epoch over the 6586 non-empty training frames
  (~263k samples seen, matching the budget KITTI baselines get from 80 epochs
  over 3712 frames).
* Anchor sizes are per-class training-set medians; `z` is the mean box-bottom
  height for that class.

### Deviations from the published recipes, and why

* **BEVFormer is replaced by PETR.** mmdetection3d does not ship BEVFormer.
  PETR is the closest available query-based multi-view camera detector and
  shares BEVFormer's premise (3D queries attending to image features).
* **PETR uses ResNet-50, not V2-99.** PETR's published V2-99 weights are a DD3D
  depth-pretrained checkpoint that is not redistributed with mmdetection3d. A
  benchmark baseline should be reproducible from public weights alone, so the
  backbone is ImageNet ResNet-50. Expect this to cost several mAP relative to
  the paper's configuration.
* **PETR trains inside the camera sector** (`ObjectAzimuthFilter`). Supervising
  it on vessels behind the boat teaches only hallucination.
* **PETR's global augmentation is reimplemented**
  (`MaritimeGlobalRotScaleTransImage`). The upstream
  `GlobalRotScaleTransImage` computes `(lidar2cam.T @ R_inv).T`, which
  *left*-multiplies `lidar2cam` and so rotates about the camera's optical axis
  rather than the world z axis. Reprojecting ground-truth corners through it
  drifts by ~470 px; through the replacement it is exact to 1e-4 px. Verify with
  `python tools/check_petr_aug.py`.
* **Depth bins reach 160 m** in BEVFusion's LSS transform, not the nuScenes
  60 m, because vessels are annotated to 178 m.
* **Camera resize preserves the full FOV.** The nuScenes recipe crops toward the
  bottom of the image; here the horizon sits near y=430 of 1080, so a bottom
  crop would discard exactly the distant vessels the benchmark is about. Resize
  factors are chosen so essentially nothing is cropped.

## 6. Results

See [RESULTS.md](RESULTS.md), regenerated by
`python tools/collect_maritime_results.py`.

### How to read them

`tools/bootstrap_maritime_ci.py` puts 95% intervals on the per-class APs by
resampling **independent encounters** with replacement -- a block bootstrap,
which is the correct unit when consecutive observations are near-duplicates of
each other. Resampling frames instead would treat 130 views of one drifting
ship as 130 independent draws and report an interval far tighter than the data
supports.

The result is blunt. On `boat`, the only class with enough encounters to
bootstrap at all, the three LiDAR baselines land at 28.8, 25.6 and 26.8 AP with
intervals roughly 19 points wide and almost entirely overlapping. **This test
split does not separate any pair of methods in the table.** The apparent
20-point spread in `mAP@3D` is carried by `ship` and `buoy`, i.e. by 1 and 4
unique objects; it is sampling noise wearing the clothes of a result.

The most visible example: BEVFusion sits ~18 mAP below TransFusion-L on test,
which looks like fusion actively hurting. It is not. BEVFusion is within 2.8
points of TransFusion-L on `boat`, *ahead* of it on `sailboat`, and beat it on
the val split at every one of the eight checkpoints (val mAP 27.7 vs 25.7 at
epoch 40). The whole gap comes from `ship` and `buoy` -- five objects -- and it
reverses between val and test. Do not read it as a modality finding.

Until more sequences are annotated, treat `boat AP@3D` as the only rankable
number here, and treat even that as ordering methods no more finely than its
interval allows.

## 7. Known gaps

* **Three of the four classes cannot support a class-level number.** `ship`,
  `sailboat` and `buoy` have 1, 2 and 4 independent encounters in the test split
  (§2). Their APs are reported for completeness and should not be used to rank
  methods, and neither should any mean that includes them. Only `boat` has
  enough independent support to carry a confidence interval, and even that
  interval is ~19 AP points wide (§6). Closing this needs more annotated
  sequences, not a different metric -- no reweighting creates evidence that the
  data does not contain.
* One sequence only. No cross-sequence, cross-weather, or cross-geography test.
* **The annotations are 7-DoF, so no target's pitch or roll is scored.** The
  label files carry `rot_y` and nothing after it, and every baseline regresses a
  single heading angle to match, so the benchmark is blind to a vessel heeling
  or pitching over a wave -- on a surface where the IMU measures the *observer*
  rolling at 0.70 deg RMS and pitching at 1.35 deg RMS (§4). This is the gap a
  maritime benchmark most needs to close and the one that costs most: it needs
  9-DoF re-annotation of all 91766 boxes and detection heads that predict nine
  numbers, not seven. Until then, treat every AP here as conditional on the
  targets being upright. `AOE` and the sea-state breakdown bound how wrong that
  is; they do not repair it. [EULER_ANGLES.md](EULER_ANGLES.md) quantifies the
  cost -- 5 deg of unmodelled target pitch would drop 25% of the `boat` and 37%
  of the `ship` boxes that currently clear IoU 0.5 -- and argues the
  re-annotation should start with `ship` pitch and `sailboat` roll rather than
  with all four classes.
* The camera-only protocol has no attitude compensation either. PETR's 3D
  position encoding is built from a static `lidar2cam`, so it assumes the
  cameras are level in every frame.
* No velocity annotations, so `code_size` is 8 rather than the 10 the nuScenes
  configs use, and no tracking or velocity metric is reported.
* The stereo pair is used as two independent monocular views. Nothing here
  exploits the 0.84 m baseline explicitly.
