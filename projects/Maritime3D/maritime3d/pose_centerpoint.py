"""CenterPoint with per-object attitude and a class-balanced heatmap loss.

Ported from the other workstation's 9-DoF CenterPoint (MaritimeCenterHead,
MaritimeCenterPointBBoxCoder, MaritimeClassBalancedGaussianFocalLoss). The
heads and parameter names match that code (``pitch`` / ``roll`` sin-cos
heads), so its checkpoints load; the decoded box follows this repo's column
order ``[x, y, z, l, w, h, yaw, pitch, roll]`` (the port emitted roll,
pitch), which is what MaritimePoseBoxes and MaritimeMetric expect.
Unlike the port, stock mmdet3d CenterHead is left untouched.
"""
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from mmdet.models.losses.gaussian_focal_loss import gaussian_focal_loss
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.task_modules.coders.centerpoint_bbox_coders import \
    CenterPointBBoxCoder
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import LiDARInstance3DBoxes


@MODELS.register_module()
class MaritimeClassBalancedGaussianFocalLoss(nn.Module):
    """Gaussian focal loss with per-class effective-number weights.

    Cui et al., "Class-Balanced Loss Based on Effective Number of Samples":
    w_c = (1 - beta) / (1 - beta^n_c), rescaled so that the box-count
    weighted mean is 1. The weight multiplies the object region (target > 0)
    of the class's heatmap channel; background pixels get ``bg_weight``.

    Args:
        num_classes (int): Number of classes.
        class_freq (Sequence[float] | None): Training box count per class,
            in label order. None disables balancing.
        beta (float): Effective-number factor in [0, 1).
        gamma (float): Negative-sample power (stock CenterPoint: 4).
        alpha (float): Prediction power (stock CenterPoint: 2).
        bg_weight (float): Weight of background pixels.
        loss_weight (float): Final loss weight.
    """

    def __init__(self,
                 num_classes: int = 5,
                 class_freq: Optional[Sequence[float]] = None,
                 beta: float = 0.9999,
                 gamma: float = 4.0,
                 alpha: float = 2.0,
                 bg_weight: float = 1.0,
                 loss_weight: float = 1.0) -> None:
        super().__init__()
        if class_freq is None:
            class_weight = torch.ones(num_classes)
        else:
            assert len(class_freq) == num_classes, \
                f'class_freq has {len(class_freq)} entries, ' \
                f'expected {num_classes}'
            freq = torch.as_tensor(class_freq, dtype=torch.float32)
            assert (freq > 0).all(), 'class_freq values must be positive'
            weight = (1.0 - beta) / (1.0 - beta**freq).clamp_min(1e-8)
            class_weight = weight / ((weight * freq).sum() / freq.sum())
        self.register_buffer('class_weight', class_weight)
        self.bg_weight = float(bg_weight)
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.loss_weight = float(loss_weight)

    def forward(self,
                pred: Tensor,
                target: Tensor,
                class_ids: Union[int, Sequence[int], None] = None,
                weight: Optional[Tensor] = None,
                avg_factor=None) -> Tensor:
        """pred / target: [B, C, H, W]; class_ids: label of each channel."""
        if class_ids is None:
            cw = pred.new_ones(1, target.shape[1], 1, 1)
        else:
            if isinstance(class_ids, int):
                class_ids = [class_ids]
            cw = self.class_weight[list(class_ids)].to(pred.dtype).view(
                1, -1, 1, 1)
        elem_weight = torch.where(target > 0, cw.expand_as(target),
                                  torch.full_like(target, self.bg_weight))
        if weight is not None:
            elem_weight = elem_weight * weight.to(pred.dtype)
        return self.loss_weight * gaussian_focal_loss(
            pred,
            target,
            elem_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='mean',
            avg_factor=avg_factor)


@TASK_UTILS.register_module()
class MaritimePoseCenterPointBBoxCoder(CenterPointBBoxCoder):
    """CenterPointBBoxCoder whose 'vel' input is the attitude code.

    ``att`` is [B, 4, H, W] = (sin, cos) of pitch then roll, and the decoded
    box is [x, y, z, l, w, h, yaw, pitch, roll]. It is passed in the slot
    stock CenterHead.predict_by_feat reserves for velocity, so the stock
    NMS / merge path is reused unchanged.
    """

    def decode(self,
               heat: Tensor,
               rot_sine: Tensor,
               rot_cosine: Tensor,
               hei: Tensor,
               dim: Tensor,
               att: Tensor,
               reg: Optional[Tensor] = None,
               task_id: int = -1) -> List[Dict[str, Tensor]]:
        batch, cat, _, _ = heat.size()
        K = self.max_num
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        def gather(x, c):
            return self._transpose_and_gather_feat(x, inds).view(batch, K, c)

        if reg is not None:
            reg = gather(reg, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5

        rot = torch.atan2(gather(rot_sine, 1), gather(rot_cosine, 1))
        att = gather(att, 4)
        pitch = torch.atan2(att[..., 0:1], att[..., 1:2])
        roll = torch.atan2(att[..., 2:3], att[..., 3:4])
        hei = gather(hei, 1)
        dim = gather(dim, 3)
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)

        xs = xs * self.out_size_factor * self.voxel_size[0] + self.pc_range[0]
        ys = ys * self.out_size_factor * self.voxel_size[1] + self.pc_range[1]
        boxes = torch.cat([xs, ys, hei, dim, rot, pitch, roll], dim=2)

        assert self.post_center_range is not None, \
            'only post_center_range is not None is supported'
        pcr = torch.as_tensor(self.post_center_range, device=heat.device)
        mask = (boxes[..., :3] >= pcr[:3]).all(2)
        mask &= (boxes[..., :3] <= pcr[3:]).all(2)
        if self.score_threshold:
            mask &= scores > self.score_threshold
        return [
            dict(bboxes=boxes[i, mask[i]],
                 scores=scores[i, mask[i]],
                 labels=clses[i, mask[i]]) for i in range(batch)
        ]


@MODELS.register_module()
class MaritimeCenterHead(CenterHead):
    """CenterHead + (pitch, roll) sin/cos heads + class-balanced heatmap.

    Needs ``common_heads`` with ``pitch=(2, 2), roll=(2, 2)``, 12
    ``code_weights`` (reg 2, height 1, dim 3, yaw 2, pitch 2, roll 2), a
    :class:`MaritimePoseCenterPointBBoxCoder` with ``code_size=9`` and 9-d
    GT boxes (dataset ``box_3d_dof=9``, ``box_type_3d='Maritime'``). With
    :class:`MaritimeClassBalancedGaussianFocalLoss` each task's heatmap is
    weighted by its classes' effective-number weights; any other heatmap
    loss is called as in CenterHead.
    """

    ATT_HEADS = ('pitch', 'roll')  # box columns 7, 8

    def get_targets_single(self, gt_instances_3d: InstanceData):
        """CenterHead targets with (sin, cos) of pitch and roll appended."""
        labels = gt_instances_3d.labels_3d
        boxes = gt_instances_3d.bboxes_3d.tensor
        base = InstanceData(
            labels_3d=labels,
            bboxes_3d=LiDARInstance3DBoxes(boxes[:, :7], box_dim=7))
        heatmaps, anno_boxes, inds, masks = super().get_targets_single(base)

        att = boxes[:, 7:9].to(labels.device)
        code = torch.stack(
            [att[:, 0].sin(), att[:, 0].cos(), att[:, 1].sin(),
             att[:, 1].cos()],
            dim=1)
        flag = 0
        for t, names in enumerate(self.class_names):
            # CenterHead orders a task's boxes class by class, in label order
            c = torch.cat(
                [code[labels == flag + j] for j in range(len(names))])
            flag += len(names)
            rows = anno_boxes[t].shape[0]
            ext = anno_boxes[t].new_zeros(rows, 4)
            ext[:min(len(c), rows)] = c[:rows]
            anno_boxes[t] = torch.cat([anno_boxes[t], ext], dim=1)
        return heatmaps, anno_boxes, inds, masks

    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args,
                     **kwargs):
        heatmaps, anno_boxes, inds, masks = self.get_targets(
            batch_gt_instances_3d)
        balanced = isinstance(self.loss_cls,
                              MaritimeClassBalancedGaussianFocalLoss)
        loss_dict = dict()
        flag = 0
        for task_id, preds_dict in enumerate(preds_dicts):
            p = preds_dict[0]
            p['heatmap'] = clip_sigmoid(p['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            n_cls = self.num_classes[task_id]
            extra = dict(class_ids=list(range(flag, flag + n_cls))) \
                if balanced else dict()
            flag += n_cls
            loss_heatmap = self.loss_cls(
                p['heatmap'],
                heatmaps[task_id],
                avg_factor=max(num_pos, 1),
                **extra)

            target_box = anno_boxes[task_id]
            pred = torch.cat([
                p[k] for k in ('reg', 'height', 'dim', 'rot') + self.ATT_HEADS
            ],
                             dim=1)
            pred = pred.permute(0, 2, 3, 1).contiguous()
            pred = self._gather_feat(
                pred.view(pred.size(0), -1, pred.size(3)), inds[task_id])
            num = masks[task_id].float().sum()
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            mask *= (~torch.isnan(target_box)).float()
            bbox_weights = mask * mask.new_tensor(
                self.train_cfg['code_weights'])
            loss_bbox = self.loss_bbox(
                pred, target_box, bbox_weights, avg_factor=(num + 1e-4))
            loss_dict[f'task{task_id}.loss_heatmap'] = loss_heatmap
            loss_dict[f'task{task_id}.loss_bbox'] = loss_bbox
        return loss_dict

    def predict_by_feat(self, preds_dicts, batch_input_metas, *args,
                        **kwargs):
        for preds_dict in preds_dicts:
            p = preds_dict[0]
            # decoded by MaritimePoseCenterPointBBoxCoder from the 'vel' slot
            p['vel'] = torch.cat([p[k] for k in self.ATT_HEADS], dim=1)
        return super().predict_by_feat(preds_dicts, batch_input_metas, *args,
                                       **kwargs)
