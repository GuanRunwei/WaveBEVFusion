"""9-DoF LiDAR boxes whose columns 7/8 are the target's pitch and roll.

Ported from the CenterPoint 9-DoF branch of the other workstation, with two
changes:

* column order follows this repo's infos and MaritimeMetric,
  ``[x, y, z_bottom, l, w, h, yaw, pitch, roll]`` (the port had roll, pitch);
* a BEV flip negates roll only. With R = Rz(yaw) Ry(pitch) Rx(roll), the
  mirrored pose is S R S' for reflections S, S' in the world / object frame,
  which leaves Ry untouched and inverts Rx (checked numerically against
  scipy: pitch -> -pitch is off by up to 0.6 rad). Intuitively, a mirrored
  bow-up hull is still bow-up, while a list to port becomes one to starboard.

Stock :class:`LiDARInstance3DBoxes` treats columns 7/8 of a 9-d box as a
velocity vector: ``rotate`` spins them with the scene and ``scale`` scales
them, which would corrupt pitch/roll under the usual augmentation.
"""
from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from mmdet3d.structures import LiDARInstance3DBoxes
from mmdet3d.structures.bbox_3d.utils import rotation_3d_in_axis
from mmdet3d.structures.points import BasePoints

PITCH, ROLL = 7, 8


class MaritimePoseBoxes(LiDARInstance3DBoxes):
    """LiDAR boxes carrying target attitude in columns 7 (pitch), 8 (roll).

    Under a rotation about the LiDAR z-axis only yaw changes (Rz is the
    outermost factor of the ZYX decomposition), scaling leaves angles alone,
    and either BEV flip negates roll and keeps pitch.
    """

    def rotate(
        self,
        angle: Union[Tensor, np.ndarray, float],
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tuple[Tensor, Tensor], Tuple[np.ndarray, np.ndarray],
               Tuple[BasePoints, Tensor], None]:
        """LiDARInstance3DBoxes.rotate without the velocity rotation."""
        if not isinstance(angle, Tensor):
            angle = self.tensor.new_tensor(angle)

        assert angle.shape == torch.Size([3, 3]) or angle.numel() == 1, \
            f'invalid rotation angle shape {angle.shape}'

        if angle.numel() == 1:
            self.tensor[:, 0:3], rot_mat_T = rotation_3d_in_axis(
                self.tensor[:, 0:3],
                angle,
                axis=self.YAW_AXIS,
                return_mat=True)
        else:
            rot_mat_T = angle
            rot_sin = rot_mat_T[0, 1]
            rot_cos = rot_mat_T[0, 0]
            angle = np.arctan2(rot_sin, rot_cos)
            self.tensor[:, 0:3] = self.tensor[:, 0:3] @ rot_mat_T

        self.tensor[:, 6] += angle

        if points is not None:
            if isinstance(points, Tensor):
                points[:, :3] = points[:, :3] @ rot_mat_T
            elif isinstance(points, np.ndarray):
                rot_mat_T = rot_mat_T.cpu().numpy()
                points[:, :3] = np.dot(points[:, :3], rot_mat_T)
            elif isinstance(points, BasePoints):
                points.rotate(rot_mat_T)
            else:
                raise ValueError
            return points, rot_mat_T

    def scale(self, scale_factor: float) -> None:
        """Scale centre and size only; pitch/roll are angles."""
        self.tensor[:, :6] *= scale_factor

    def flip(
        self,
        bev_direction: str = 'horizontal',
        points: Optional[Union[Tensor, np.ndarray, BasePoints]] = None
    ) -> Union[Tensor, np.ndarray, BasePoints, None]:
        """BEV flip: yaw as in LiDARInstance3DBoxes, roll negated, pitch
        kept."""
        assert bev_direction in ('horizontal', 'vertical')
        if bev_direction == 'horizontal':
            self.tensor[:, 1] = -self.tensor[:, 1]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6]
        else:
            self.tensor[:, 0] = -self.tensor[:, 0]
            if self.with_yaw:
                self.tensor[:, 6] = -self.tensor[:, 6] + np.pi
        if self.box_dim > ROLL:
            self.tensor[:, ROLL] = -self.tensor[:, ROLL]

        if points is not None:
            assert isinstance(points, (Tensor, np.ndarray, BasePoints))
            if isinstance(points, (Tensor, np.ndarray)):
                if bev_direction == 'horizontal':
                    points[:, 1] = -points[:, 1]
                else:
                    points[:, 0] = -points[:, 0]
            elif isinstance(points, BasePoints):
                points.flip(bev_direction)
            return points
