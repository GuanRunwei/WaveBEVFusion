# Euler angles in a maritime 3D detection benchmark

What pitch, yaw and roll do to detection on water, what this dataset can and
cannot measure about them, and what a 9-DoF re-annotation should prioritise if
one is ever funded.

Reproduce every number here with:

```bash
python tools/extract_ego_attitude.py           # observer attitude, from the IMU
python tools/analyze_attitude_sensitivity.py \
       --preds <DumpResults pkl>               # target attitude, geometry + headroom
```

---

## 1. The premise, and why it is only half right

The motivating observation is correct: a road is rigid and the sea is not, so a
driving benchmark's assumption that the sensor rig stays level does not carry
over. But "rough water needs Euler angles, calm water does not" is the wrong
decomposition, because it merges two effects with different physics, different
error scaling, and different remedies.

|  | observer attitude | target attitude |
|---|---|---|
| whose angles | the vessel carrying the sensors | each detected vessel |
| error scales with | tilt **x range** | tilt **x the target's own size** |
| worst case | small tilt, distant target | large tilt, large target |
| measured here? | yes, IMU at 96.4 Hz | **no, not annotated** |
| fix | de-rotate the cloud (needs an IMU) | 9-DoF boxes (needs re-annotation *and* new heads) |

Conflating them produces the wrong engineering conclusion in both directions:
it suggests calm water is safe when it is not, and it suggests 9-DoF boxes are
the fix when for the observer term a preprocessing step is.

---

## 2. Observer attitude: always matters, never a detection problem

The IMU at `dataset/00/lidar_front/imu.txt` gives 213781 samples at 96.4 Hz.
After a complementary filter (§5 of the main README explains why neither the
gyro nor the accelerometer works alone):

| | RMS | dominant period | extremes |
|---|---|---|---|
| roll | 0.70 deg | 2.25 s (wind waves) | -8.3 .. +4.9 deg |
| pitch | 1.35 deg | 19.0 s (swell) | -3.2 .. +3.9 deg |

**This is a calm record.** Sub-degree RMS roll is not a seaway anyone would
describe as rough. And yet:

| tilt | displacement at 10 m | at 50 m | at 150 m |
|---|---|---|---|
| 0.7 deg | 0.12 m | 0.61 m | 1.83 m |
| 1.4 deg | 0.24 m | 1.22 m | 3.67 m |

IoU 0.5 on a 6 m boat allows about 2 m of total centre error. At 150 m the
*median* attitude of this calm record already spends the entire budget. So the
honest answer to "is calm water fine" is **no** -- but the reason is range, not
sea state. A driving benchmark can ignore rig attitude because its objects sit
inside 20 m, not because tarmac is flat. Move the same 1 deg out to maritime
range and it dominates.

The remedy, though, is not a richer box. It is to de-rotate the point cloud
with the IMU before the network ever sees it. That is a solved preprocessing
step whenever an IMU is co-located with the LiDAR, as it is here.

> **Correction (2026-09-26): the clouds are NOT levelled.** A raw-scan test
> on the organised Ouster buffers (each beam's elevation stays within
> 0.016-0.06 deg across frames while IMU pitch spans 0.3-3.9 deg; slope vs
> IMU 0.00-0.06, levelling would give 1.0) shows `points4` is in the raw,
> body-fixed sensor frame, and the static `lidar2cam` projection is correct at
> every attitude (work_dirs/tmp/horizon/proj_report.txt). The sea therefore
> tilts in the LiDAR frame with the hull's attitude; the detected horizon
> tracks the IMU with slope ~1. The per-frame IMU up vector in the LiDAR frame
> is in `dataset/infos/detection/sea_up_lidar.npz`. The paragraph below is
> kept for the record.

**And it appears to have been applied already.** Ground-truth `z` scatter grows
0.88 / 1.05 / 1.33 m across the three range bands. A fully body-fixed cloud
under the attitude above would produce 0.59 / 1.77 / 3.06 m, and the correlation
between `z` residual and IMU-predicted tilt decays from +0.47 at 0-50 m to +0.06
at 100-160 m. Some levelling is upstream of the data as distributed.

Which is exactly why the sea-state breakdown in
[RESULTS.md](RESULTS.md) comes out null -- `boat` AP moves -0.70 / +5.11 /
-2.16 / +3.25 points from the calm band to the rough one across the four
baselines, non-monotonically and with inconsistent sign. There is little
residual observer-attitude sensitivity left to measure. That is a statement
about this sequence's preprocessing, not a finding that hull motion is
harmless.

### The mistake this cost us

The first version of that breakdown banded frames by **tilt magnitude** and
reported a 10-23 point AP drop from calm to rough for every method. It was
wrong, and worth recording as a trap.

An accelerometer cannot distinguish a roll from the centripetal acceleration of
a sustained turn. A two-minute turn therefore reads as two minutes of heel; on
this record the tilt estimate correlates +0.30 with sustained yaw rate, and the
top tilt tercile turns at 1.28 deg/s against 0.63 deg/s in the bottom. Turning
is not independent of what is in the scene, so the bands ended up holding
different populations:

| tilt-banded (**discarded**) | calm | rough |
|---|---|---|
| boat boxes | 11233 | **194** |
| median boat range | 84 m | **14 m** |

Comparing those two sets says which vessels happened to be nearby, not what
waves do. The fix is to band on the **RMS of the roll+pitch rate** over a 4 s
window: a steady turn is near-DC in rate and a static trim offset is exactly DC,
so both drop out and only wave-driven oscillation survives. The resulting bands
are matched on everything that would otherwise explain a gap -- median boat
range 82 / 81 / 78 m, 33 / 27 / 37 per cent beyond 100 m, spread over 21 / 26 /
23 temporal blocks. General lesson: **when conditioning a metric on a nuisance
variable, check that the conditioning did not also sort the scene content.**

---

## 3. Target attitude: what a 9-DoF box would buy

This is the part the annotations cannot answer. The KITTI-format labels carry
`rot_y` and a constant `-1` after it -- yaw only -- and all five baselines
regress a single heading angle by construction. So per-object pitch and roll
error is unmeasurable on this data, and no metric here reports it.

What *is* computable is the price of ignoring them. A 7-DoF detector emits a box
with zero pitch and zero roll. If the target is genuinely tilted, the best that
detector can score -- perfect centre, perfect extent, perfect yaw -- is the IoU
between the tilted box and its un-tilted twin. That ceiling follows from the
class median dimensions alone.

### Pitch matters more than roll, which is the opposite of the intuition

A box is a prism, so pitch turns it in the (length, height) plane and roll in
the (width, height) plane. The aspect ratio of *that* plane decides everything:

| class | dx x dy x dz (m) | pitch-plane ratio | roll-plane ratio | dangerous angle |
|---|---|---|---|---|
| boat | 17.11 x 4.27 x 6.08 | 2.81 | 0.70 | **pitch** |
| ship | 76.65 x 10.96 x 20.08 | 3.82 | 0.55 | **pitch** |
| sailboat | 14.61 x 4.03 x 13.96 | 1.05 | 0.29 | **roll** |
| buoy | 2.49 x 2.58 x 6.49 | 0.38 | 0.40 | neither |

Everyone pictures a boat rolling. But hulls are long and flat, so the roll plane
is nearly square and rotating in it barely changes the overlap, while the pitch
plane is elongated and punishes rotation hard. IoU ceilings, and the angle at
which each ceiling reaches the class's evaluation threshold:

| class | pitch 2 deg | 5 deg | 10 deg | pitch crit | roll 2 deg | 5 deg | 10 deg | roll crit |
|---|---|---|---|---|---|---|---|---|
| boat | 0.947 | 0.877 | 0.777 | 29.8 deg | 0.965 | 0.918 | 0.854 | never |
| ship | 0.932 | 0.842 | 0.716 | **21.0 deg** | 0.960 | 0.908 | 0.835 | 54.6 deg |
| sailboat | 0.967 | 0.923 | 0.863 | never | 0.938 | 0.855 | 0.737 | **23.4 deg** |
| buoy | 0.950 | 0.884 | 0.790 | 73.6 deg | 0.952 | 0.887 | 0.795 | 83.6 deg |

`never` means the box is close enough to square in that plane that no rotation
alone can push it below threshold -- `boat` cannot be lost to roll at any angle.

The sailboat inverts because of its mast: 14.0 m tall against 4.0 m of beam
makes it the one tall-and-narrow class, and the only one for which roll is the
angle to worry about.

### It is a tax on the margin, not a cliff

Critical angles of 21-30 deg look reassuring: no vessel heels that far, and this
record's own hull peaked at 8.3 deg of roll and 3.9 deg of pitch. Read that way,
7-DoF looks adequate.

That reading is wrong, because a detector does not reach the threshold with its
error budget intact -- it has already spent most of it on centre and extent.
Applying the ceiling to PointPillars' measured IoU distribution, the share of
ground-truth boxes that **currently clear the threshold and would stop clearing
it** if the target were tilted:

| class | GT | passing | pitch 2 deg | 5 deg | 10 deg | roll 2 deg | 5 deg | 10 deg |
|---|---|---|---|---|---|---|---|---|
| boat | 14067 | 5194 | 9.8% | **25.5%** | 52.8% | 6.4% | 16.3% | 31.8% |
| ship | 128 | 79 | 12.7% | **36.7%** | 59.5% | 8.9% | 19.0% | 39.2% |
| sailboat | 875 | 300 | 3.0% | 7.7% | 14.7% | 6.3% | 16.7% | **79.0%** |
| buoy | 275 | 267 | 0.4% | 3.4% | 11.2% | 0.4% | 3.0% | 10.9% |

Five degrees of unmodelled pitch -- well inside what this calm record's own hull
experienced -- removes a quarter of the passing `boat` boxes and over a third of
`ship`, against a nominal critical angle of 21-30 deg. The gap between those two
readings *is* the point.

### A partial natural cancellation, and where it fails

Small craft respond strongly to 2 s wind waves; a 77 m ship, whose natural
period is 10-15 s, barely notices them. So the large angles land on the small
targets, which are also the ones with the least to lose in IoU. That cancellation
is real and it is why 7-DoF survives at sea better than the geometry alone
suggests.

It fails in one specific combination: **a large vessel in swell.** Long-period
energy is what a big hull does respond to -- and this record's pitch is
swell-driven at a 19.0 s period, squarely in that band. `ship` is simultaneously
the most pitch-sensitive class (critical angle 21.0 deg, the lowest in the
table) and the one whose resonance the dominant sea state actually excites.

---

## 4. What is scored instead

Yaw is annotated, so yaw is scored: **AOE**, mean absolute heading error over
detections matched within 10 m of BEV centre distance. Matched on distance
rather than IoU deliberately -- matching on IoU would measure heading only where
the box is already good, which is precisely where heading is least interesting.

| method | mAOE | mAOE180 |
|---|---|---|
| PointPillars | 44.5 | 15.7 |
| SECOND | 51.1 | 12.1 |
| TransFusion-L | **26.3** | **9.3** |
| BEVFusion | 41.1 | 12.7 |

`AOE` counts a bow/stern swap as a 180 deg error; `AOE180` folds to [0, 90] and
does not. The gap between the columns is large for **every** method, which says
the dominant heading failure is not a skewed box but a flipped one: the axis is
recovered and the direction along it is not. A vessel is close to symmetric in a
sparse point cloud at 80 m, so this is unsurprising -- but it is separable and
therefore actionable, which an aggregate heading error would have hidden.

---

## 5. Conclusions

1. **"Calm water is fine" is false, and range is why.** At 150 m this record's
   median attitude already consumes the entire IoU 0.5 budget for a 6 m boat.
   The variable is tilt x range, not sea state.
2. **Observer attitude is a preprocessing problem, not a detection problem.**
   With a co-located IMU, de-rotate the cloud. This dataset's producer appears
   to have done so, which is why the sea-state breakdown is null.
3. **Target attitude is where 9-DoF would pay, and pitch dominates roll.** Hulls
   are long and flat; the roll plane is nearly square and cheap, the pitch plane
   is elongated and expensive. `boat` cannot be lost to roll at any angle.
4. **The sailboat is the exception**, and only because of its mast.
5. **The effect is a margin tax.** Critical angles say 21-30 deg and look safe;
   measured against real detector output, 5 deg of pitch costs a quarter of
   `boat` and a third of `ship`.
6. **The bad combination is a large vessel in swell**, which is exactly the
   regime this record's 19 s pitch period sits in.

### If a 9-DoF re-annotation is funded

Do not re-annotate all 91766 boxes. In priority order:

1. **`ship` pitch** -- lowest critical angle (21.0 deg), highest headroom loss
   (36.7% at 5 deg), and the class whose resonance the dominant swell excites.
2. **`sailboat` roll** -- 79.0% loss at 10 deg, and the only class where roll is
   the dangerous angle.
3. **`boat` pitch** -- individually cheaper, but 51447 training boxes make it
   the only class with enough support to actually train a 9-DoF head on.
4. **`buoy`** -- skip. Near-isotropic and evaluated at IoU 0.25; the loss stays
   in single digits out to 10 deg.

Note that this needs annotation **and** detection heads that emit nine numbers.
Half of it is useless: 9-DoF labels scored against 7-DoF predictions just move
the ceiling into the ground truth.

### Limits of everything above

* **No target vessel's attitude is measured anywhere in this dataset.** Every
  angle in §3 is a counterfactual -- "if a vessel were heeled this far, this is
  what it would cost" -- not an observed distribution. The observer's IMU is
  used as a stand-in for what a similar-sized hull experiences in the same
  seaway, which is an assumption, not a measurement.
* The headroom table applies the ceiling multiplicatively to measured IoU,
  assuming tilt error is independent of the centre and extent error already
  present. Read it as an order of magnitude.
* One sequence, one vessel, one sea state, and a calm one. Sub-degree RMS roll
  is not a seaway; nothing here bounds behaviour at Beaufort 5 and above, where
  both terms grow and the cancellation in §3 stops helping.
