"""Matching costs specific to the maritime3d benchmark."""
from typing import Optional, Sequence

import torch

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class NormalizedBBox3DL1Cost:
    """L1 matching cost with box centres normalised by ``pc_range``.

    PETR's ``BBox3DL1Cost`` takes the L1 distance in the raw target space, and
    ``normalize_bbox`` leaves cx/cy/cz there in **metres**. At nuScenes'
    +-51.2 m that keeps the regression cost within an order of magnitude of the
    classification cost, so both terms influence the Hungarian assignment.

    At the +-160 m this benchmark needs, the centre terms grow until they swamp
    everything else. Measured on this data at initialisation, the spread of
    ``reg_cost`` across candidate pairs is 25.7 against 0.139 for ``cls_cost``
    -- matching is decided 185:1 by geometry alone. The classification head then
    has no influence on which query is assigned to an object, never receives a
    consistent gradient, and collapses to a constant score (every query in every
    frame scored 0.1011 after 40 epochs, with ``loss_cls`` flat at 1.19 from the
    first epoch to the last). AP is then exactly 0 no matter how good the boxes
    are, which is what the first PETR run produced.

    Normalising the centres to the unit cube makes the cost balance independent
    of detection range, restoring the proportions PETR was tuned at.

    Args:
        weight (float): Scale factor for the whole cost.
        pc_range (Sequence[float]): ``[x0, y0, z0, x1, y1, z1]`` used to
            normalise the centre dimensions. When ``None`` this degrades to the
            plain unnormalised L1 cost.
    """

    # normalize_bbox packs boxes as
    # (cx, cy, log dx, log dy, cz, log dz, sin rot, cos rot[, vx, vy])
    _CX, _CY, _CZ = 0, 1, 4

    def __init__(self,
                 weight: float = 1.0,
                 pc_range: Optional[Sequence[float]] = None) -> None:
        self.weight = weight
        self.pc_range = None if pc_range is None else list(pc_range)

    def _scale(self, bbox_pred: torch.Tensor) -> torch.Tensor:
        scale = bbox_pred.new_ones(bbox_pred.shape[-1])
        if self.pc_range is None:
            return scale
        r = self.pc_range
        for dim, (lo, hi) in zip((self._CX, self._CY, self._CZ),
                                 ((r[0], r[3]), (r[1], r[4]), (r[2], r[5]))):
            if dim < scale.numel():
                scale[dim] = 1.0 / max(hi - lo, 1e-6)
        return scale

    def __call__(self, bbox_pred: torch.Tensor,
                 gt_bboxes: torch.Tensor) -> torch.Tensor:
        scale = self._scale(bbox_pred)
        cost = torch.cdist(bbox_pred * scale, gt_bboxes * scale, p=1)
        return cost * self.weight

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(weight={self.weight}, '
                f'pc_range={self.pc_range})')
