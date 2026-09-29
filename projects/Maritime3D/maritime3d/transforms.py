"""Transforms specific to the maritime3d benchmark."""
from typing import Sequence

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class StripEmptyLidarPoints(BaseTransform):
    """Drop capture-buffer padding rows from the raw lidar bins.

    The maritime capture writes every scan into a fixed 131072x7 float32
    buffer; only the rows that received a return hold real coordinates --
    the rest are all-zero rows scattered through the file (a real return
    never sits exactly at the sensor origin, and water-surface points near
    the vessel still have |x|+|y|+|z| > 0). Loading with load_dim=4 without
    this filter trains on ~75% fake origin points, so every pipeline must
    load with load_dim=7 and strip the padding before any voxel/range
    filtering.

    Args:
        tol (float): rows with |x|+|y|+|z| <= tol are padding. Defaults to
            1e-6 m.
    """

    def __init__(self, tol: float = 1e-6) -> None:
        self.tol = float(tol)

    def transform(self, input_dict: dict) -> dict:
        points = input_dict['points']
        xyz = points.tensor[:, :3]
        keep = xyz.abs().sum(dim=1) > self.tol
        input_dict['points'] = points[keep]
        return input_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(tol={self.tol})'


@TRANSFORMS.register_module()
class ObjectAzimuthFilter(BaseTransform):
    """Drop ground-truth boxes outside an azimuth sector.

    The LiDAR is annotated over the full circle, but the two stereo cameras
    only span ~59 deg ahead of the vessel. Supervising a camera-only detector
    on objects behind it teaches nothing except to hallucinate, so the
    camera-only baseline trains inside its own field of view. LiDAR and fusion
    baselines do not use this transform.

    Args:
        azimuth_range (tuple): ``(lo, hi)`` bearings in degrees, measured from
            +x (vessel heading) toward +y.
    """

    def __init__(self, azimuth_range: Sequence[float]) -> None:
        self.azimuth_range = (float(azimuth_range[0]), float(azimuth_range[1]))

    def transform(self, input_dict: dict) -> dict:
        boxes = input_dict['gt_bboxes_3d']
        centers = boxes.gravity_center.numpy()
        az = np.degrees(np.arctan2(centers[:, 1], centers[:, 0]))
        lo, hi = self.azimuth_range
        mask = (az >= lo) & (az <= hi)
        # np.bool_ rather than a torch mask, matching ObjectRangeFilter
        input_dict['gt_bboxes_3d'] = boxes[mask]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][mask]
        return input_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(azimuth_range={self.azimuth_range})'


@TRANSFORMS.register_module()
class MaritimeGlobalRotScaleTransImage(BaseTransform):
    """Global rotate + scale for camera-only training, without touching images.

    Rotates and scales the world, then folds the inverse into ``lidar2cam`` so
    that every 3D point still projects to the pixel it came from. Written from
    scratch rather than subclassing PETR's ``GlobalRotScaleTransImage``, which
    is not self-consistent on this data: its ``rotate_bev_along_z`` computes
    ``(lidar2cam.T @ R_inv).T``, i.e. it left-multiplies ``lidar2cam``, which
    rotates about the *camera's* optical axis instead of the world z axis.
    Reprojecting ground-truth corners through it drifts by several hundred
    pixels; through this one it is exact to 1e-4 px (see the check in
    ``tools/check_petr_aug.py``).

    Args:
        rot_range (list): Yaw range in radians.
        scale_ratio_range (list): Uniform scale range.
        translation_std (list): Unused, kept for config compatibility.
        training (bool): When False the transform is a no-op.
    """

    def __init__(self,
                 rot_range: Sequence[float] = (-0.3925, 0.3925),
                 scale_ratio_range: Sequence[float] = (0.95, 1.05),
                 translation_std: Sequence[float] = (0, 0, 0),
                 training: bool = True) -> None:
        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std
        self.training = training

    def transform(self, results: dict) -> dict:
        if not self.training:
            return results

        angle = float(np.random.uniform(*self.rot_range))
        scale = float(np.random.uniform(*self.scale_ratio_range))

        # LiDARInstance3DBoxes.rotate(a) maps p -> R(a) p and yaw -> yaw + a
        results['gt_bboxes_3d'].rotate(np.array(angle))
        results['gt_bboxes_3d'].scale(scale)

        c, s = np.cos(angle), np.sin(angle)
        world = np.eye(4)
        world[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0],
                                  [0.0, 0.0, 1.0]]) * scale
        world_inv = np.linalg.inv(world)

        lidar2cam = np.asarray(results['lidar2cam'], dtype=np.float64)
        results['lidar2cam'] = lidar2cam @ world_inv
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(rot_range={self.rot_range}, '
                f'scale_ratio_range={self.scale_ratio_range}, '
                f'training={self.training})')
