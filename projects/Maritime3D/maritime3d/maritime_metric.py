# Copyright (c) OpenMMLab. All rights reserved.
"""Detection metric for the maritime3d benchmark.

Reports KITTI-style 40-point interpolated AP at both 3D and BEV IoU, plus a
breakdown by range. Unlike the KITTI metric this does not need 2D boxes or
difficulty levels, and it keeps frames with no ground truth so that false
positives on empty water are penalised.
"""
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmcv.ops import box_iou_rotated
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from terminaltables import AsciiTable

from mmdet3d.registry import METRICS
from mmdet3d.structures import LiDARInstance3DBoxes

DEFAULT_RANGES = ((0, 50), (50, 100), (100, 160))
# BEV centre-distance thresholds, in metres, for the distance-based AP.
#
# IoU 0.5 on a 6 m boat demands ~2 m of centre accuracy, which no camera-only
# method reaches at 50-160 m: the stereo pair has a 0.84 m baseline, worth
# 9-15 px of disparity out there, so every AP@IoU it produces is exactly 0 and
# the metric stops distinguishing "learned nothing" from "off by 10 m". This is
# the same reason nuScenes scores on centre distance rather than IoU. The
# ladder is nuScenes' 0.5/1/2/4 m scaled to a domain whose median object sits
# at 77 m instead of 20 m.
DEFAULT_DIST_THRESHOLDS = (2.0, 5.0, 10.0, 20.0)
# Matching threshold, in metres, for the orientation error. Yaw error is only
# defined on matched pairs, and matching on IoU would define it exactly where
# the box is already good -- a detector that is 10 m off with the heading right
# would contribute nothing. 10 m is loose enough to admit those cases and tight
# enough that a match is still the same vessel.
DEFAULT_ORIENT_MATCH_DIST = 10.0
# Sea state, as the RMS of the hull's roll+pitch *rate* over a 4 s window, in
# deg/s -- how hard the platform is rocking, not how far it is leaning.
#
# Banding on tilt is the obvious choice and is wrong. An accelerometer cannot
# separate a roll from the centripetal acceleration of a sustained turn, so a
# long turn reads as sustained heel; on this record tilt correlates +0.30 with
# sustained yaw rate. Turning is not independent of what is in the scene, and a
# tilt-banded test split ends up with 11233 boat boxes at a median 84 m in calm
# against 194 at a median 14 m in rough -- a different population, so an AP
# gap between the bands is about which vessels were nearby, not about waves.
#
# A rate RMS has neither problem: a steady turn is near-DC in roll and pitch
# rate and a static trim offset is exactly DC, so both drop out and only
# wave-driven oscillation survives. The resulting test bands are matched on
# everything that would otherwise explain a gap -- median boat range 82 / 81 /
# 78 m, 33 / 27 / 37 per cent of boats beyond 100 m, spread over 21 / 26 / 23
# temporal blocks.
#
# The edges are the terciles of the full annotated record of sequence 00, fixed
# as constants rather than recomputed per split so the bands stay comparable as
# sequences are added.
DEFAULT_SEA_STATES = (('calm', 0.0, 0.77), ('moderate', 0.77, 1.48),
                      ('rough', 1.48, float('inf')))


def _iou(pred: np.ndarray, gt: np.ndarray, mode: str) -> np.ndarray:
    """Pairwise similarity between [N, 7] and [M, 7] LiDAR boxes.

    ``mode='3d'`` and ``'bev'`` return IoU. ``mode='dist'`` returns the
    *negated* BEV centre distance, so that "higher is better" holds for all
    three and the greedy matcher in :func:`_eval_class` needs no special case
    -- a distance criterion of ``d`` metres is then just a threshold of ``-d``.
    """
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros((len(pred), len(gt)), dtype=np.float32)
    if mode == 'dist':
        d = np.hypot(pred[:, None, 0] - gt[None, :, 0],
                     pred[:, None, 1] - gt[None, :, 1])
        return (-d).astype(np.float32)
    # box_dim must be explicit: the constructor does not infer it from the
    # tensor width and would assert 7-dim on 9-DoF (pitch/roll) arrays
    a = LiDARInstance3DBoxes(
        torch.as_tensor(pred, dtype=torch.float32), box_dim=pred.shape[-1])
    b = LiDARInstance3DBoxes(
        torch.as_tensor(gt, dtype=torch.float32), box_dim=gt.shape[-1])
    with torch.no_grad():
        if mode == '3d':
            return a.overlaps(a, b).cpu().numpy()
        # same path as BaseInstance3DBoxes.overlaps takes for its BEV term:
        # .bev is (x, y, dx, dy, yaw), which is what box_iou_rotated wants.
        # Clamp the extents to keep the CUDA kernel from overflowing.
        a_bev, b_bev = a.bev, b.bev
        a_bev[:, 2:4] = a_bev[:, 2:4].clamp(min=1e-4)
        b_bev[:, 2:4] = b_bev[:, 2:4].clamp(min=1e-4)
        return box_iou_rotated(a_bev, b_bev).cpu().numpy()


def _ap40(rec: np.ndarray, prec: np.ndarray) -> float:
    """40-point interpolated AP, as used by the KITTI leaderboard."""
    if len(rec) == 0:
        return 0.0
    ap = 0.0
    for t in np.linspace(1 / 40, 1.0, 40):
        p = prec[rec >= t]
        ap += (p.max() if len(p) else 0.0)
    return float(ap / 40)


def _eval_class(preds: List[dict], thr: float, mode: str) -> float:
    """AP for one class at one IoU threshold.

    Each entry of ``preds`` is one frame: ``boxes``/``scores`` for detections
    and ``gt`` for ground truth, already filtered to the class and range of
    interest. Detections are matched to ground truth greedily in score order,
    one ground truth per detection.
    """
    n_gt = sum(len(f['gt']) for f in preds)
    if n_gt == 0:
        return float('nan')

    scores, tps = [], []
    for f in preds:
        box, sc, gt = f['boxes'], f['scores'], f['gt']
        if not len(box):
            continue
        order = np.argsort(-sc)
        box, sc = box[order], sc[order]
        ious = _iou(box, gt, mode)
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(box)):
            hit = False
            if len(gt):
                # -inf, not -1: in 'dist' mode the matrix holds negated metres,
                # so -1 would advertise an already-taken box as a 1 m match
                cand = np.where(~taken, ious[i], -np.inf)
                j = int(cand.argmax())
                if cand[j] >= thr:
                    taken[j] = True
                    hit = True
            scores.append(sc[i])
            tps.append(hit)
    if not scores:
        return 0.0

    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    return _ap40(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9))


def _yaw_errors(preds: List[dict], dist: float) -> np.ndarray:
    """Absolute yaw error, in degrees, over centre-distance-matched pairs.

    Water is not a road: the hull rolls and pitches, so a vessel's apparent
    heading in the sensor frame is the true heading plus a projection of the
    platform's own attitude. Detections are matched greedily in score order,
    exactly as :func:`_eval_class` does, but on centre distance rather than
    IoU, so that a well-oriented box with a poor centre still contributes.

    See :func:`_angle_errors` for the matching and wrap rules.
    """
    return _angle_errors(preds, dist, (6, ))[0]


def _angle_errors(preds: List[dict], dist: float,
                  dims: Sequence[int]) -> np.ndarray:
    """Absolute Euler-angle error, in degrees, over centre-distance-matched
    pairs, one array per requested box column.

    Detections are matched greedily in score order, exactly as
    :func:`_eval_class` does, but on centre distance rather than IoU, so that
    a well-oriented box with a poor centre still contributes. Column 6 (yaw)
    wraps to ``[-pi, pi]`` before going unsigned; the attitude columns 7/8
    (pitch, roll) are small excursions around 0 and are compared directly.

    Returns an array of shape ``(len(dims), n_pairs)`` -- columns share the
    same matches, so per-axis errors stay comparable row by row.
    """
    out = [[] for _ in dims]
    for f in preds:
        box, sc, gt = f['boxes'], f['scores'], f['gt']
        if not len(box) or not len(gt):
            continue
        # A column is comparable only when BOTH sides carry it: 9-DoF
        # runs score yaw, pitch and roll over the same matches, 7-DoF
        # runs carry 7-wide boxes and score yaw alone. Gating on
        # max(dims) wholesale dropped yaw for 7-DoF baselines too --
        # every overall AOE came out nan while the per-range yaw
        # columns still had matches.
        have = {
            d
            for d in dims if box.shape[-1] >= d + 1 and gt.shape[-1] >= d + 1
        }
        if not have:
            continue
        order = np.argsort(-sc)
        box = box[order]
        neg_d = _iou(box, gt, 'dist')
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(box)):
            cand = np.where(~taken, neg_d[i], -np.inf)
            j = int(cand.argmax())
            if cand[j] >= -dist:
                taken[j] = True
                for k, d in enumerate(dims):
                    if d in have:
                        out[k].append(box[i, d] - gt[j, d])
    errs = []
    for k, d in enumerate(dims):
        if not out[k]:
            errs.append(np.zeros(0, dtype=np.float64))
            continue
        e = np.asarray(out[k], dtype=np.float64)
        if d == 6:
            e = np.abs((e + np.pi) % (2 * np.pi) - np.pi)
        else:
            e = np.abs(e)
        errs.append(e)
    return errs


def _eval_class_ignore(preds: List[dict], thr: float, mode: str) -> float:
    """AP with KITTI-style "don't care" handling.

    Each frame carries ``gt_ign`` (GT outside the evaluated subset) and
    ``det_ign`` (detection whose own attribute is outside the subset).
    A detection matched to an ignored GT, or unmatched but itself outside the
    subset, is neither TP nor FP; ignored GT never counts as a miss. Matching
    tries in-subset GT first.
    """
    n_gt = sum(int((~f['gt_ign']).sum()) for f in preds)
    if n_gt == 0:
        return float('nan')
    scores, tps = [], []
    for f in preds:
        box, sc, gt = f['boxes'], f['scores'], f['gt']
        if not len(box):
            continue
        order = np.argsort(-sc)
        box, sc, dign = box[order], sc[order], f['det_ign'][order]
        ious = _iou(box, gt, mode)
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(box)):
            state = 'fp'
            if len(gt):
                for want_ign in (False, True):
                    m = (~taken) & (f['gt_ign'] == want_ign)
                    cand = np.where(m, ious[i], -np.inf)
                    j = int(cand.argmax())
                    if cand[j] >= thr:
                        taken[j] = True
                        state = 'ign' if want_ign else 'tp'
                        break
            if state == 'fp' and dign[i]:
                state = 'ign'
            if state == 'ign':
                continue
            scores.append(sc[i])
            tps.append(state == 'tp')
    if not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray(tps, dtype=np.float64)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    return _ap40(ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9))


def _matched_errors(preds: List[dict], dist: float) -> np.ndarray:
    """[n, 3] (|dxy| centre error, |dz| bottom error, gt length) over pairs
    matched greedily in score order within ``dist`` metres (BEV)."""
    out = []
    for f in preds:
        box, sc, gt = f['boxes'], f['scores'], f['gt']
        if not len(box) or not len(gt):
            continue
        order = np.argsort(-sc)
        box = box[order]
        neg_d = _iou(box, gt, 'dist')
        taken = np.zeros(len(gt), dtype=bool)
        for i in range(len(box)):
            cand = np.where(~taken, neg_d[i], -np.inf)
            j = int(cand.argmax())
            if cand[j] >= -dist:
                taken[j] = True
                out.append((-cand[j], abs(box[i, 2] - gt[j, 2]), gt[j, 3]))
    return np.asarray(out, dtype=np.float64).reshape(-1, 3)


@METRICS.register_module()
class MaritimeMetric(BaseMetric):
    """3D / BEV AP for the maritime3d benchmark.

    Args:
        iou_thresholds (dict): Per-class IoU thresholds. Large vessels use a
            looser threshold than the KITTI car setting because annotation of
            hull extent is inherently coarser at 100 m+.
        ranges (tuple): Range bands, in metres, reported separately.
        pcd_limit_range (list): Boxes outside this range are ignored.
        azimuth_range (tuple, optional): ``(lo, hi)`` bearings in degrees,
            measured from +x (vessel heading) toward +y. When set, only boxes
            whose centre falls in that sector are evaluated -- both predictions
            and ground truth, so boxes the sensor cannot see are neither missed
            detections nor false positives. Used to score the camera-only
            baseline over the region its cameras actually cover; leave unset
            for the full-circle protocol every other baseline is scored on.
        ego_attitude (str, optional): Path to the ``ego_attitude.npz`` written
            by ``tools/extract_ego_attitude.py``. When present, AP is
            additionally broken down by sea state -- how hard the hull was
            rocking when the frame was taken. Silently skipped if the file is
            absent, so the benchmark still runs on a copy of the data without
            the IMU log.
        sea_states (tuple): ``(name, lo, hi)`` rocking-rate bands, in deg/s.
        orient_match_dist (float): Centre distance, in metres, at which a
            detection is matched to ground truth for the yaw error.
    """

    def __init__(self,
                 iou_thresholds: Optional[Dict[str, float]] = None,
                 ranges: Sequence[Sequence[float]] = DEFAULT_RANGES,
                 dist_thresholds: Sequence[float] = DEFAULT_DIST_THRESHOLDS,
                 pcd_limit_range: Sequence[float] = (-160.0, -160.0, -8.0,
                                                     160.0, 160.0, 24.0),
                 azimuth_range: Optional[Sequence[float]] = None,
                 ego_attitude: Optional[str] = 'data/maritime/ego_attitude.npz',
                 sea_states: Sequence[Sequence] = DEFAULT_SEA_STATES,
                 orient_match_dist: float = DEFAULT_ORIENT_MATCH_DIST,
                 info_file: Optional[str] = None,
                 length_bins: Sequence[Sequence[float]] = ((0, 15), (15, 30),
                                                           (30, 1e9)),
                 err_match_dist: float = 5.0,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = 'Maritime',
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        self.azimuth_range = (None if azimuth_range is None else
                              (float(azimuth_range[0]),
                               float(azimuth_range[1])))
        self.iou_thresholds = iou_thresholds or {
            'boat': 0.5,
            'ship': 0.5,
            'sailboat': 0.5,
            'buoy': 0.25,
        }
        self.ranges = tuple(tuple(r) for r in ranges)
        self.dist_thresholds = tuple(float(d) for d in dist_thresholds)
        self.pcd_limit_range = np.asarray(pcd_limit_range, dtype=np.float32)
        self.sea_states = tuple(tuple(s) for s in sea_states)
        self.orient_match_dist = float(orient_match_dist)
        self.rock = self._load_attitude(ego_attitude)
        # Optional benchmark breakdowns (camera FOV, day/night, vessel
        # length, ATE-z); enabled by pointing info_file at the split's info
        # pkl, which supplies per-frame calibration and day/night.
        self.frame_meta = self._load_frame_meta(info_file)
        self.length_bins = tuple(tuple(b) for b in length_bins)
        self.err_match_dist = float(err_match_dist)

    @staticmethod
    def _load_attitude(path: Optional[str]) -> Optional[Dict[str, float]]:
        """``{timestamp: rocking rate}``, deg/s of RMS roll+pitch rate.

        Keyed on the LiDAR filename stem -- the capture time in nanoseconds --
        and not on ``sample_idx``, because mmengine's
        ``BaseDataset.get_data_info`` overwrites ``sample_idx`` with the
        frame's position in the split and discards the value in the info pkl.
        Keying on it looked up the wrong frames while still finding a hit for
        every one, since the positions 0..N are valid indices elsewhere in the
        record; the bad join was visible only as a sea-state histogram that
        disagreed with the one computed directly from the npz.

        The npz also carries ``roll`` and ``pitch``, and banding on those is
        the tempting mistake: see ``DEFAULT_SEA_STATES`` for why a tilt-based
        split confounds sea state with turning and produces bands that hold
        different vessels rather than the same vessels in different conditions.
        """
        if not path or not os.path.exists(path):
            return None
        z = np.load(path)
        return dict(zip([str(t) for t in z['token']], z['rock'].tolist()))

    @staticmethod
    def _token(sample: dict) -> Optional[str]:
        """Capture timestamp of a frame, from its LiDAR path."""
        path = sample.get('lidar_path')
        return None if not path else os.path.splitext(
            os.path.basename(path))[0]

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for sample in data_samples:
            pred = sample['pred_instances_3d']
            gt = sample['eval_ann_info']
            gt_boxes = gt['gt_bboxes_3d']
            if not isinstance(gt_boxes, np.ndarray):
                gt_boxes = gt_boxes.tensor.numpy()
            # (n, dof), dof=7 or 9. Reshape with both dims explicit: the
            # len/`-1` form fails on empty frames -- numpy cannot infer an
            # unknown dim from zero elements.
            gt_boxes = np.ascontiguousarray(
                gt_boxes, dtype=np.float32).reshape(gt_boxes.shape[0],
                                                    gt_boxes.shape[1])
            self.results.append(
                dict(
                    token=self._token(sample),
                    boxes=pred['bboxes_3d'].tensor.cpu().numpy(),
                    scores=pred['scores_3d'].cpu().numpy(),
                    labels=pred['labels_3d'].cpu().numpy(),
                    gt_boxes=gt_boxes,
                    gt_labels=np.asarray(gt['gt_labels_3d'],
                                         dtype=np.int64).reshape(-1)))

    def _in_range(self, boxes: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0, dtype=bool)
        r = np.hypot(boxes[:, 0], boxes[:, 1])
        return (r >= lo) & (r < hi)

    def _in_limit(self, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros(0, dtype=bool)
        lo, hi = self.pcd_limit_range[:3], self.pcd_limit_range[3:]
        keep = ((boxes[:, :3] >= lo) & (boxes[:, :3] <= hi)).all(1)
        if self.azimuth_range is not None:
            az = np.degrees(np.arctan2(boxes[:, 1], boxes[:, 0]))
            keep &= (az >= self.azimuth_range[0]) & (az <=
                                                     self.azimuth_range[1])
        return keep

    def compute_metrics(self, results: list) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()
        classes = self.dataset_meta['classes']

        # drop boxes outside the evaluated volume once, up front
        frames = []
        for r in results:
            pm = self._in_limit(r['boxes'])
            gm = self._in_limit(r['gt_boxes'])
            frames.append(
                dict(
                    token=r.get('token'),
                    boxes=r['boxes'][pm],
                    scores=r['scores'][pm],
                    labels=r['labels'][pm],
                    gt_boxes=r['gt_boxes'][gm],
                    gt_labels=r['gt_labels'][gm]))

        def select(ci, band=None, keep=None):
            out = []
            for f in frames:
                if keep is not None and not keep(f):
                    continue
                pm = f['labels'] == ci
                gm = f['gt_labels'] == ci
                pb, ps = f['boxes'][pm], f['scores'][pm]
                gb = f['gt_boxes'][gm]
                if band is not None:
                    pb2 = self._in_range(pb, *band)
                    pb, ps = pb[pb2], ps[pb2]
                    gb = gb[self._in_range(gb, *band)]
                out.append(dict(boxes=pb, scores=ps, gt=gb))
            return out

        metrics: Dict[str, float] = {}
        table = [['class', 'n_gt', 'AP@3D', 'AP@BEV'] +
                 [f'3D {lo}-{hi}m' for lo, hi in self.ranges]]
        ap3d, apbev = [], []
        for ci, name in enumerate(classes):
            thr = self.iou_thresholds.get(name, 0.5)
            per = select(ci)
            n_gt = sum(len(f['gt']) for f in per)
            a3 = _eval_class(per, thr, '3d')
            ab = _eval_class(per, thr, 'bev')
            row = [name, str(n_gt), f'{a3 * 100:.2f}', f'{ab * 100:.2f}']
            metrics[f'{name}_AP3D'] = a3
            metrics[f'{name}_APBEV'] = ab
            for band in self.ranges:
                a = _eval_class(select(ci, band), thr, '3d')
                metrics[f'{name}_AP3D_{band[0]}-{band[1]}m'] = a
                row.append('-' if np.isnan(a) else f'{a * 100:.2f}')
            table.append(row)
            if not np.isnan(a3):
                ap3d.append(a3)
                apbev.append(ab)

        metrics['mAP3D'] = float(np.mean(ap3d)) if ap3d else float('nan')
        metrics['mAPBEV'] = float(np.mean(apbev)) if apbev else float('nan')
        table.append([
            'mAP', '', f"{metrics['mAP3D'] * 100:.2f}",
            f"{metrics['mAPBEV'] * 100:.2f}"
        ] + [''] * len(self.ranges))

        print_log(
            '\nMaritime3D detection results (AP40, IoU ' + ', '.join(
                f'{k}={v}' for k, v in self.iou_thresholds.items()) + ')\n' +
            AsciiTable(table).table,
            logger=logger)

        # Distance-based AP. Reported alongside, never instead of, the IoU
        # numbers: it ignores extent and orientation entirely, so it answers
        # "is there a vessel about there" rather than "is this box right".
        dtable = [['class', 'n_gt'] +
                  [f'AP@{d:g}m' for d in self.dist_thresholds] + ['mean']]
        per_class_mean = []
        for ci, name in enumerate(classes):
            per = select(ci)
            n_gt = sum(len(f['gt']) for f in per)
            aps = [_eval_class(per, -d, 'dist') for d in self.dist_thresholds]
            for d, a in zip(self.dist_thresholds, aps):
                metrics[f'{name}_APdist_{d:g}m'] = a
            m = float(np.mean(aps)) if not np.isnan(aps[0]) else float('nan')
            metrics[f'{name}_APdist'] = m
            dtable.append([name, str(n_gt)] +
                          ['-' if np.isnan(a) else f'{a * 100:.2f}'
                           for a in aps] +
                          ['-' if np.isnan(m) else f'{m * 100:.2f}'])
            if not np.isnan(m):
                per_class_mean.append(m)
        metrics['mAPdist'] = (float(np.mean(per_class_mean))
                              if per_class_mean else float('nan'))
        dtable.append(['mAP', ''] + [''] * len(self.dist_thresholds) +
                      [f"{metrics['mAPdist'] * 100:.2f}"])
        print_log(
            '\nMaritime3D detection results (AP40, BEV centre distance)\n' +
            AsciiTable(dtable).table,
            logger=logger)

        self._report_orientation(metrics, classes, select, logger)
        self._report_sea_state(metrics, classes, select, frames, logger)
        if self.frame_meta is not None:
            self._report_breakdowns(metrics, classes, frames, logger)
        return metrics

    def _report_orientation(self, metrics, classes, select, logger) -> None:
        """Euler-angle errors over centre-distance-matched pairs.

        Yaw (box column 6) is scored for every baseline. Pitch and roll
        (columns 7/8) are scored when BOTH the predictions and the ground
        truth carry them -- i.e. for 10-DoF baselines on the 10-DoF
        annotation; for a 7-DoF baseline the metrics are reported as nan
        rather than dropped, so tables line up across baselines.

        Two yaw columns, because they fail differently. ``AOE`` wraps to
        ``[0, 180]`` and so counts a bow/stern swap as a 180 deg error;
        ``AOE180`` folds the angle to ``[0, 90]`` and does not. A large gap
        between them means the heading axis is right and the direction along it
        is not, which is a different defect from a box that is simply skewed.
        Pitch/roll are small excursions around 0, so a single unwrapped
        absolute error is the whole story -- there is no wrap ambiguity to
        fold, hence no second column.
        """
        d = self.orient_match_dist
        table = [['class', 'n_match', 'AOE', 'AOE180', 'AOE_pitch',
                  'AOE_roll'] + [f'AOE {lo}-{hi}m' for lo, hi in self.ranges]]
        aoe_all, pitch_all, roll_all = [], [], []
        for ci, name in enumerate(classes):
            ey, ep, er = _angle_errors(select(ci), d, (6, 7, 8))
            fold = np.minimum(ey, np.pi - ey)
            if len(ey):
                a, a180 = np.degrees(ey.mean()), np.degrees(fold.mean())
                metrics[f'{name}_AOE'] = float(a)
                metrics[f'{name}_AOE180'] = float(a180)
                aoe_all.append(a)
                row = [name, str(len(ey)), f'{a:.2f}', f'{a180:.2f}']
            else:
                metrics[f'{name}_AOE'] = float('nan')
                metrics[f'{name}_AOE180'] = float('nan')
                row = [name, '0', '-', '-']
            # attitude errors over the SAME matched pairs as yaw
            for key, e, acc in (('pitch', ep, pitch_all),
                                ('roll', er, roll_all)):
                if len(e):
                    v = float(np.degrees(e.mean()))
                    metrics[f'{name}_AOE_{key}'] = v
                    acc.append(v)
                else:
                    metrics[f'{name}_AOE_{key}'] = float('nan')
                row.append(f'{metrics[f"{name}_AOE_{key}"]:.2f}'
                           if len(e) else '-')
            for band in self.ranges:
                eb = _yaw_errors(select(ci, band), d)
                v = float(np.degrees(eb.mean())) if len(eb) else float('nan')
                metrics[f'{name}_AOE_{band[0]}-{band[1]}m'] = v
                row.append('-' if np.isnan(v) else f'{v:.2f}')
            table.append(row)
        metrics['mAOE'] = float(np.mean(aoe_all)) if aoe_all else float('nan')
        metrics['mAOE_pitch'] = (float(np.mean(pitch_all))
                                 if pitch_all else float('nan'))
        metrics['mAOE_roll'] = (float(np.mean(roll_all))
                                if roll_all else float('nan'))
        table.append([
            'mean', '', f"{metrics['mAOE']:.2f}", '',
            f"{metrics['mAOE_pitch']:.2f}" if pitch_all else '-',
            f"{metrics['mAOE_roll']:.2f}" if roll_all else '-'
        ] + [''] * len(self.ranges))
        print_log(
            f'\nMaritime3D orientation error (deg, matched within {d:g} m)\n' +
            AsciiTable(table).table,
            logger=logger)

    def _report_sea_state(self, metrics, classes, select, frames,
                          logger) -> None:
        """AP@3D split by how hard the hull was rocking in each frame.

        This is the part of the Euler-angle question the data can actually
        answer. Per-object pitch and roll are unannotated, but the *platform's*
        are recorded at 96 Hz, and they are what turn a level-ground assumption
        into a bad one: the hull rolls at 0.70 deg RMS on a 2.25 s period and
        pitches at 1.35 deg RMS on a 19 s swell, and one degree of tilt
        displaces a target 2.6 m at 150 m against an IoU-0.5 error budget of
        about 2 m for a 6 m boat.

        The bands are cut on rate rather than tilt so that they hold the same
        vessels in different conditions rather than different vessels; see
        ``DEFAULT_SEA_STATES``. A method that holds its AP across them is
        robust to sea state on this recording; one that does not has been
        reading a flat-water prior.
        """
        if self.rock is None:
            return
        known = [f for f in frames if f.get('token') in self.rock]
        if len(known) < 0.5 * len(frames):
            print_log(
                f'sea-state breakdown skipped: attitude known for only '
                f'{len(known)}/{len(frames)} frames',
                logger=logger)
            return

        def band_of(f):
            return self.rock.get(f.get('token'))

        table = [['class'] + [f'{n} [{lo:g},{hi:g}) deg/s'
                              for n, lo, hi in self.sea_states]]
        counts = []
        for n, lo, hi in self.sea_states:
            counts.append(sum(1 for f in known if lo <= band_of(f) < hi))
        per_state = {n: [] for n, _, _ in self.sea_states}
        for ci, name in enumerate(classes):
            thr = self.iou_thresholds.get(name, 0.5)
            row = [name]
            for n, lo, hi in self.sea_states:
                def keep(f, lo=lo, hi=hi):
                    t = band_of(f)
                    return t is not None and lo <= t < hi

                a = _eval_class(select(ci, keep=keep), thr, '3d')
                metrics[f'{name}_AP3D_{n}'] = a
                row.append('-' if np.isnan(a) else f'{a * 100:.2f}')
                if not np.isnan(a):
                    per_state[n].append(a)
            table.append(row)
        row = ['mAP']
        for n, _, _ in self.sea_states:
            m = float(np.mean(per_state[n])) if per_state[n] else float('nan')
            metrics[f'mAP3D_{n}'] = m
            row.append('-' if np.isnan(m) else f'{m * 100:.2f}')
        table.append(row)
        table.append(['frames'] + [str(c) for c in counts])
        print_log(
            '\nMaritime3D AP@3D by sea state (RMS hull roll+pitch rate)\n'
            + AsciiTable(table).table,
            logger=logger)

    @staticmethod
    def _load_frame_meta(path: Optional[str]) -> Optional[dict]:
        """``{timestamp: (daynight, [3x4 lidar2img per camera], (w, h))}``
        from an info pkl written by tools/create_maritime_infos_from_tables.
        Keyed like ``_token``: the LiDAR file stem."""
        if not path:
            return None
        import pickle
        with open(path, 'rb') as f:
            infos = pickle.load(f)['data_list']
        meta = {}
        for info in infos:
            tok = os.path.splitext(
                os.path.basename(info['lidar_points']['lidar_path']))[0]
            cams = [(np.asarray(c['lidar2img'], dtype=np.float64)[:3],
                     c['width'], c['height'])
                    for c in info.get('images', {}).values()]
            meta[tok] = (info.get('daynight', 'unknown'), cams)
        return meta

    def _in_fov(self, token, boxes: np.ndarray) -> np.ndarray:
        """Box centre projects into any camera of the frame."""
        if len(boxes) == 0 or token not in self.frame_meta:
            return np.zeros(len(boxes), dtype=bool)
        c = np.c_[boxes[:, 0], boxes[:, 1], boxes[:, 2] + 0.5 * boxes[:, 5],
                  np.ones(len(boxes))]
        hit = np.zeros(len(boxes), dtype=bool)
        for P, w, h in self.frame_meta[token][1]:
            q = c @ P.T
            ok = q[:, 2] > 0.5
            u = np.where(ok, q[:, 0] / np.where(ok, q[:, 2], 1), -1)
            v = np.where(ok, q[:, 1] / np.where(ok, q[:, 2], 1), -1)
            hit |= ok & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        return hit

    def _report_breakdowns(self, metrics, classes, frames, logger) -> None:
        """Benchmark breakdowns (projects/Maritime3D/MARIFUSION_CL.md §6).

        * fov / nonfov: box centre inside / outside both camera images.
          Predictions and GT are filtered alike, so boxes on the other side
          count neither as misses nor as false positives.
        * day / night: frame-level, from the info pkl.
        * vessel length bins: KITTI-style ignore (see _eval_class_ignore),
          testing VWA, whose gain should sit in long vessels.
        * ATE-xy / ATE-z over centre-matched pairs and the BEV-minus-3D AP
          gap, testing SSR, whose gain should show up as z quality.
        """
        nan = float('nan')

        def fmt(v, pct=True):
            return '-' if np.isnan(v) else (f'{v * 100:.2f}'
                                            if pct else f'{v:.2f}')

        for f in frames:
            f['_pfov'] = self._in_fov(f['token'], f['boxes'])
            f['_gfov'] = self._in_fov(f['token'], f['gt_boxes'])
            m = self.frame_meta.get(f['token'])
            f['_dn'] = m[0] if m else 'unknown'

        def subset(ci, pmask, gmask, keep=None):
            out = []
            for f in frames:
                if keep is not None and not keep(f):
                    continue
                pm = (f['labels'] == ci) & pmask(f)
                gm = (f['gt_labels'] == ci) & gmask(f)
                out.append(
                    dict(
                        boxes=f['boxes'][pm],
                        scores=f['scores'][pm],
                        gt=f['gt_boxes'][gm]))
            return out

        def all_p(f):
            return np.ones(len(f['boxes']), dtype=bool)

        def all_g(f):
            return np.ones(len(f['gt_boxes']), dtype=bool)

        def mean_ap(sel, evaluator):
            a3, ab, per = [], [], {}
            for ci, name in enumerate(classes):
                thr = self.iou_thresholds.get(name, 0.5)
                p = sel(ci)
                x3, xb = evaluator(p, thr, '3d'), evaluator(p, thr, 'bev')
                per[name] = x3
                if not np.isnan(x3):
                    a3.append(x3)
                    ab.append(xb)
            return (float(np.mean(a3)) if a3 else nan,
                    float(np.mean(ab)) if ab else nan, per)

        groups = [
            ('fov', lambda ci: subset(ci, lambda f: f['_pfov'],
                                      lambda f: f['_gfov'])),
            ('nonfov', lambda ci: subset(ci, lambda f: ~f['_pfov'],
                                         lambda f: ~f['_gfov'])),
            ('day', lambda ci: subset(ci, all_p, all_g,
                                      lambda f: f['_dn'] == 'day')),
            ('night', lambda ci: subset(ci, all_p, all_g,
                                        lambda f: f['_dn'] == 'night')),
        ]
        for lo, hi in self.length_bins:
            tag = f'len{lo:g}-{hi:g}m' if hi < 1e8 else f'len>{lo:g}m'

            def sel(ci, lo=lo, hi=hi):
                out = []
                for f in frames:
                    pm, gm = f['labels'] == ci, f['gt_labels'] == ci
                    pb, gb = f['boxes'][pm], f['gt_boxes'][gm]
                    out.append(
                        dict(
                            boxes=pb,
                            scores=f['scores'][pm],
                            gt=gb,
                            det_ign=~((pb[:, 3] >= lo) & (pb[:, 3] < hi)),
                            gt_ign=~((gb[:, 3] >= lo) & (gb[:, 3] < hi))))
                return out

            groups.append((tag, sel))

        table = [['subset', 'mAP3D', 'mAPBEV', 'BEV-3D'] + list(classes)]
        for name, sel in groups:
            ev = _eval_class_ignore if name.startswith('len') else _eval_class
            m3, mb, per = mean_ap(sel, ev)
            metrics[f'mAP3D_{name}'] = m3
            metrics[f'mAPBEV_{name}'] = mb
            for cname, v in per.items():
                metrics[f'{cname}_AP3D_{name}'] = v
            table.append([name, fmt(m3), fmt(mb), fmt(mb - m3)] +
                         [fmt(per[c]) for c in classes])
        metrics['gap_BEV_3D'] = metrics['mAPBEV'] - metrics['mAP3D']
        print_log(
            '\nMaritime3D breakdowns (AP40; length bins use KITTI-style '
            'ignore)\n' + AsciiTable(table).table,
            logger=logger)

        # translation errors over centre-matched pairs, pooled over classes
        errs = [
            _matched_errors(subset(ci, all_p, all_g), self.err_match_dist)
            for ci in range(len(classes))
        ]
        e = np.concatenate(errs, 0) if errs else np.zeros((0, 3))
        etable = [['pairs', 'n', 'ATE-xy mean', 'ATE-z mean', 'ATE-z median']]
        bins = [('all', np.ones(len(e), dtype=bool))] + [
            (f'len{lo:g}-{hi:g}m' if hi < 1e8 else f'len>{lo:g}m',
             (e[:, 2] >= lo) & (e[:, 2] < hi)) for lo, hi in self.length_bins
        ]
        for name, m in bins:
            if m.any():
                xy, z = float(e[m, 0].mean()), float(e[m, 1].mean())
                zmed = float(np.median(e[m, 1]))
            else:
                xy = z = zmed = nan
            sfx = '' if name == 'all' else f'_{name}'
            metrics[f'ATE_xy{sfx}'] = xy
            metrics[f'ATE_z{sfx}'] = z
            metrics[f'ATE_z_median{sfx}'] = zmed
            etable.append([name, str(int(m.sum())), fmt(xy, False),
                           fmt(z, False), fmt(zmed, False)])
        print_log(
            f'\nMaritime3D translation error (m, matched within '
            f'{self.err_match_dist:g} m)\n' + AsciiTable(etable).table,
            logger=logger)
