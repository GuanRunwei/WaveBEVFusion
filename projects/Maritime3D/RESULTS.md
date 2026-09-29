# maritime3d benchmark results

Sequence 00, held-out test split (3884 frames / 15354 objects).
AP40 at IoU 0.5 for boat / ship / sailboat and 0.25 for buoy.

| Method | Modality | mAP@3D | mAP@BEV | boat @3D | ship @3D | sailboat @3D | buoy @3D |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointPillars | LiDAR | 31.46 | 37.64 | 28.81 | 52.06 | 27.92 | 17.06 |
| SECOND | LiDAR | 25.49 | 32.52 | 25.62 | 27.43 | 25.33 | 23.56 |
| TransFusion-L | LiDAR | 43.35 | 51.47 | 26.80 | 46.75 | 35.12 | 64.73 |
| BEVFusion | LiDAR + stereo camera | 25.10 | 31.32 | 23.99 | 27.39 | 36.01 | 13.02 |
| PETR | stereo camera | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

> **Do not rank methods on `mAP@3D`.** It averages four classes whose independent
> support differs by an order of magnitude: the test split holds 12 boat encounters
> but only 1 ship, 2 sailboat and 4 buoy. Three quarters of the mean therefore
> rests on seven unique objects, and `ship AP` is the answer to "was this one vessel
> found". See the confidence intervals below and README section 2.

## Confidence intervals

95% intervals from `tools/bootstrap_maritime_ci.py`, which resamples
**independent object encounters** with replacement, not frames. At
10 Hz the 130 test `ship` boxes are one vessel drifting 0.10 m per
frame; treating them as 130 draws would understate the interval by
roughly the square root of that duplication.

| Method | boat (12 enc) | ship (1 enc) | sailboat (2 enc) | buoy (4 enc) |
| --- | --- | --- | --- | --- |
| PointPillars | 28.8 [21.9, 40.7] | 52.1 (n/a) | 27.9 (n/a) | 17.1 (n/a) |
| SECOND | 25.6 [19.1, 38.0] | 27.4 (n/a) | 25.3 (n/a) | 23.6 (n/a) |
| TransFusion-L | 26.8 [19.8, 40.5] | 46.7 (n/a) | 35.1 (n/a) | 64.7 (n/a) |
| BEVFusion | 24.0 [17.4, 33.4] | 27.4 (n/a) | 36.0 (n/a) | 13.0 (n/a) |

`(n/a)` means the class has too few independent encounters for a
bootstrap to mean anything -- with 4 encounters there are only 4**4 =
256 distinct resamples, and the percentiles are an artefact of that
support rather than a statement about the method. Only `boat`
qualifies, and its interval is wide enough to overlap across every
baseline in this table. **No pair of methods here is separated by
this test split.**

## AP@3D by BEV centre distance

Matches a detection to ground truth by centre distance instead of
IoU, so extent and heading are ignored: this answers "is there a
vessel about there", not "is this box right". Reported because
IoU 0.5 on a 6 m boat demands ~2 m of centre accuracy, which a
0.84 m stereo baseline cannot deliver at 50-160 m -- every AP@IoU a
camera-only method produces is exactly 0, and a column of zeros
cannot distinguish "learned nothing" from "off by 10 m". nuScenes
uses centre distance for the same reason.

| Method | Modality | mAP@2m | mAP@5m | mAP@10m | mAP@20m | mean |
| --- | --- | --- | --- | --- | --- | --- |
| PointPillars | LiDAR | 26.40 | 32.18 | 34.88 | 47.72 | 35.29 |
| SECOND | LiDAR | 29.07 | 39.49 | 42.51 | 47.29 | 39.59 |
| TransFusion-L | LiDAR | 40.65 | 50.85 | 57.00 | 69.24 | 54.44 |
| BEVFusion | LiDAR + stereo camera | 25.93 | 37.45 | 43.92 | 49.66 | 39.24 |
| PETR | stereo camera | 0.00 | 0.02 | 0.78 | 4.37 | 1.29 |

## Orientation error (yaw)

Mean absolute heading error, in degrees, over detections matched to
ground truth within 10 m of BEV centre distance -- matching on IoU
would measure yaw only where the box is already good. `AOE` counts a
bow/stern swap as 180 deg; `AOE180` folds the angle to [0, 90] and
does not, so a gap between the two columns means the heading *axis*
is right and the direction along it is not.

Yaw is the only Euler angle scored here. See README section 4 on why
pitch and roll are not, and what that omits.

| Method | boat AOE | ship AOE | sailboat AOE | buoy AOE | mAOE | mAOE180 |
| --- | --- | --- | --- | --- | --- | --- |
| PointPillars | 32.2 | 29.2 | 66.6 | 50.0 | 44.5 | 15.7 |
| SECOND | 39.7 | 1.8 | 80.9 | 82.0 | 51.1 | 12.1 |
| TransFusion-L | 31.4 | 4.0 | 39.2 | 30.6 | 26.3 | 9.3 |
| BEVFusion | 41.9 | 24.9 | 48.6 | 49.0 | 41.1 | 12.7 |
| PETR | 99.5 | -- | 58.3 | 169.2 | 109.0 | 13.1 |

## AP@3D by sea state

Test frames banded by how hard the hull is rocking -- the RMS of its
roll+pitch rate over a 4 s window, from
`data/maritime/ego_attitude.npz`: calm < 0.77 deg/s, moderate
0.77-1.48 deg/s, rough > 1.48 deg/s (1174 / 1623 / 1087 test
frames). Edges are the terciles of the full annotated record, fixed
as constants so the bands stay comparable as sequences are added.

This is the Euler-angle question the data can answer. Per-object
pitch and roll are unannotated, but the *platform* rolls at 0.70 deg
RMS on a 2.25 s period and pitches at 1.35 deg RMS on a 19 s swell,
and that is what makes a level-ground prior wrong: one degree of
tilt displaces a target 2.6 m at 150 m, against an IoU-0.5 budget of
about 2 m for a 6 m boat.

**Rate, not tilt.** An accelerometer cannot separate a roll from the
centripetal acceleration of a sustained turn, so banding on tilt
bands partly on turning -- and turning is not independent of what is
in the scene. The tilt-banded version of this table put 11233 boat
boxes at a median 84 m in calm against 194 at a median 14 m in
rough, which compares different vessels, not different conditions.
A rate RMS drops both a steady turn and a static trim (near-DC in
rate), leaving the wave-driven oscillation, and the bands below are
matched at median boat range 82 / 81 / 78 m with 33 / 27 / 37 per
cent of boats beyond 100 m over 21 / 26 / 23 temporal blocks.

| Method | boat calm | boat moderate | boat rough | boat delta | mAP calm | mAP moderate | mAP rough |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PointPillars | 32.67 | 26.43 | 31.97 | -0.70 | 65.89 | 50.88 | 38.80 |
| SECOND | 26.35 | 24.17 | 31.46 | +5.11 | 59.55 | 36.54 | 34.85 |
| TransFusion-L | 32.46 | 22.46 | 30.30 | -2.16 | 67.49 | 50.11 | 39.61 |
| BEVFusion | 27.40 | 20.85 | 30.65 | +3.25 | 65.70 | 27.61 | 36.36 |

`boat` carries this table and the `mAP` columns should be ignored
in it. `ship` has no encounter at all in the rough band and `buoy`
none in the calm one, so their AP there is `nan` and the mean
silently becomes an average over whichever classes happened to
survive. `boat` is the only class with enough independent
encounters (12) to band three ways at all, and even it is thin: the
rough band holds 2233 boat boxes but they are not 2233 independent
observations.

**Read as a null result.** No method degrades monotonically with sea
state, the `boat delta` changes sign across methods, and all four
dip in the *moderate* band and recover in the rough one -- a shape
no motion mechanism explains, and one that points at scene
composition rather than at waves. The deltas are also far inside the
~19-point `boat` interval from the encounter bootstrap. On this
recording, residual sensitivity to hull rocking is not measurable at
this split's resolution -- which is consistent with the levelling
already applied upstream of the point cloud (README section 4), and
is a statement about this sequence, not about detectors at sea.

## AP@3D by range

| Method | boat 0-50m | boat 50-100m | boat 100-160m | ship 0-50m | ship 50-100m | ship 100-160m | sailboat 0-50m | sailboat 50-100m | sailboat 100-160m | buoy 0-50m | buoy 50-100m | buoy 100-160m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PointPillars | 32.03 | 35.95 | 12.86 | -- | 52.40 | -- | 62.30 | 28.15 | 0.00 | -- | 90.99 | 50.49 |
| SECOND | 28.77 | 32.92 | 9.35 | -- | 28.70 | -- | 73.32 | 24.40 | 0.00 | -- | 70.72 | 36.30 |
| TransFusion-L | 19.91 | 36.25 | 8.82 | -- | 49.47 | -- | 66.13 | 37.77 | 0.00 | -- | 93.77 | 54.04 |
| BEVFusion | 10.86 | 37.24 | 8.17 | -- | 30.90 | -- | 69.14 | 37.35 | 0.00 | -- | 52.75 | 27.46 |
| PETR | 0.00 | 0.00 | 0.00 | -- | 0.00 | -- | 0.00 | 0.00 | 0.00 | -- | 0.00 | 0.00 |

## Camera-sector protocol (PETR only)

PETR sees roughly [-35, +24] deg of the 360 deg the LiDAR is
annotated over, so the full-circle mAP above is bounded by the
rig, not the method. Scored inside the camera sector:

| Metric | boat | ship | sailboat | buoy | mAP |
| --- | --- | --- | --- | --- | --- |
| AP@3D | 0.00 | -- | 0.00 | 0.00 | 0.00 |
| AP@BEV | 0.00 | -- | 0.00 | 0.00 | 0.00 |
