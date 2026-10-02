"""CenterFormer (projects/CenterFormer) with the Maritime3D components.

The sea-surface supervision and MariFusion of the CenterPoint models,
carried over to CenterFormer:

* :class:`SeaCenterFormerBboxHead`: CenterFormerBboxHead with the
  waterline-uncertainty ('sig') and freeboard ('fb') branches and the
  sea-surface losses of :class:`EvidenceSeaCenterHead` (IMU tilt with a
  learned gain, Kalman mean level, Gaussian-process waves), evaluated on the
  queries that sit on a GT centre. Training supervision only: at test time
  the boxes keep the head's own heights. Its multi-class NMS covers any
  number of classes (upstream: 3).
* :class:`MaritimeCenterFormer`: CenterFormer with hard voxelisation (the
  LiDAR input of the CenterPoint models), the metas handed to the head, and
  an optional second stage: the two front cameras and the
  RaySplitFusionRefineHead of MariFusion on CenterFormer's stage-1
  detections, sampling the BEV map the heatmap is predicted on.
"""
from typing import List

import torch
from torch import Tensor

from mmdet3d.registry import MODELS
from projects.CenterFormer.centerformer import (CenterFormer,
                                                CenterFormerBboxHead)
from projects.CenterFormer.centerformer.bbox_ops import nms_iou3d
from projects.CenterFormer.centerformer.centerformer_head import \
    get_corresponding_box
from .evidence_centerpoint import EvidenceSeaCenterHead
from .mari_fusion import image_branch, refine_losses, refine_predictions
from .sea_surface import SeaSurface


@MODELS.register_module()
class SeaCenterFormerBboxHead(CenterFormerBboxHead):
    """CenterFormerBboxHead + sea-surface supervision (see module docstring).

    Args:
        train_cfg (dict): The detector's train_cfg (grid, voxel size, ...).
        sea_surface (dict, optional): :class:`SeaSurface` arguments; None
            gives the plain head (with the generalised NMS).
        loss_final_height_weight, loss_sigma_weight, loss_level_weight
            (float): As :class:`EvidenceSeaCenterHead`.
    """

    # the sea-surface losses of the CenterPoint head; they read the box code
    # [sub-cell offset (2), z, log l w h (3), sin, cos, a2c (2)] at flat
    # heatmap indices, which is CenterFormer's code plus a zero a2c
    _cell_to_m = EvidenceSeaCenterHead._cell_to_m
    _sea_losses = EvidenceSeaCenterHead._sea_losses

    def __init__(self,
                 *args,
                 train_cfg: dict = None,
                 sea_surface: dict = None,
                 loss_final_height_weight: float = 0.25,
                 loss_sigma_weight: float = 0.25,
                 loss_level_weight: float = 0.5,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.train_cfg = train_cfg
        self.norm_bbox = True  # CenterFormer regresses log dimensions
        self.a2c_scale, self.a2c_frame = 1.0, 'bev'
        self.sea = None
        if sea_surface is not None:
            heads = self.tasks[0].heads
            assert 'sig' in heads and 'fb' in heads, \
                "the sea surface needs common_heads 'sig' and 'fb'"
            self.sea = SeaSurface(**sea_surface)
        self.loss_final_height_weight = float(loss_final_height_weight)
        self.loss_sigma_weight = float(loss_sigma_weight)
        self.loss_level_weight = float(loss_level_weight)
        self._metas = None

    def loss(self, preds_dicts, example, metas: List[dict] = None,
             **kwargs) -> dict:
        losses = super().loss(preds_dicts, example, **kwargs)
        if self.sea is None:
            return losses
        at_gt = []
        for task_id, p in enumerate(preds_dicts):
            pred = p['anno_box'].transpose(1, 2)  # [B, K, 8], set by super
            tgt, sel, _ = get_corresponding_box(
                p['order'], example['ind'][task_id],
                example['mask'][task_id], example['cat'][task_id],
                example['anno_box'][task_id])
            zero = pred.new_zeros(*pred.shape[:2], 2)
            at_gt.append(dict(
                pred=torch.cat([pred, zero], -1),
                tgt=torch.cat([tgt, zero], -1), ind=p['order'],
                mask=sel.float(), score=p['scores'].detach(),
                log_b=p['sig'][:, 0], fb=p['fb'][:, 0]))
        self._metas = metas
        try:
            with torch.autocast('cuda', enabled=False):
                losses.update(self._sea_losses(at_gt))
        finally:
            self._metas = None
        return losses

    def post_processing(self, img_metas, batch_box_preds, batch_score,
                        batch_label, test_cfg, post_center_range, task_id,
                        batch_mask, batch_iou):
        """Range filter, IoU rescoring and per-class NMS (any class count;
        the thresholds and sizes of ``test_cfg.nms`` are per class)."""
        nms = test_cfg.nms
        factor = batch_score.new_tensor(self.iou_factor)
        out = []
        for i in range(len(batch_score)):
            box, s, lab = batch_box_preds[i], batch_score[i], batch_label[i]
            keep = batch_mask[i] & \
                (box[:, :3] >= post_center_range[:3]).all(1) & \
                (box[:, :3] <= post_center_range[3:]).all(1)
            box, s, lab = box[keep], s[keep], lab[keep]
            if batch_iou is not None:
                s = s * batch_iou[i][keep].pow(factor[lab])
            sel = []
            for c in range(sum(self.num_classes)):
                idx = (lab == c).nonzero()[:, 0]
                if idx.numel() == 0:
                    continue
                k = nms_iou3d(
                    box[idx][:, [0, 1, 2, 3, 4, 5, -1]].float(),
                    s[idx].float(),
                    thresh=nms.nms_iou_threshold[c],
                    pre_maxsize=nms.nms_pre_max_size[c],
                    post_max_size=nms.nms_post_max_size[c])
                sel.append(idx[k])
            sel = torch.cat(sel) if sel else lab.new_zeros(0)
            out.append(dict(bboxes=box[sel], scores=s[sel],
                            labels=lab[sel]))
        return out


@MODELS.register_module()
class MaritimeCenterFormer(CenterFormer):
    """CenterFormer for Maritime3D (see module docstring).

    Args:
        img_backbone, img_neck (dict, optional): Camera branch of the
            second stage.
        refine_head (dict, optional): RaySplitFusionRefineHead; None gives
            the one-stage detector.
    """

    def __init__(self,
                 *args,
                 img_backbone: dict = None,
                 img_neck: dict = None,
                 refine_head: dict = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_backbone = MODELS.build(img_backbone) \
            if img_backbone else None
        self.img_neck = MODELS.build(img_neck) if img_neck else None
        self.refine_head = MODELS.build(refine_head) if refine_head else None
        # the BEV map the heatmap is predicted on (input of hm_head), for
        # the second stage
        self._bev = None
        self.backbone.hm_head.register_forward_pre_hook(self._grab_bev)

    def _grab_bev(self, module, inputs):
        if self.refine_head is not None:
            self._bev = inputs[0]

    def init_weights(self) -> None:
        super().init_weights()  # CenterFormer's: default init + BN weights
        for m in (self.img_backbone, self.img_neck, self.refine_head):
            if m is not None:
                m.init_weights()

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_input_metas: List[dict]) -> Tensor:
        v = batch_inputs_dict['voxels']
        feats = self.voxel_encoder(v['voxels'], v['num_points'], v['coors'])
        batch_size = v['coors'][-1, 0].item() + 1
        return self.middle_encoder(feats, v['coors'], batch_size)

    def _images(self, batch_inputs_dict):
        imgs = batch_inputs_dict.get('imgs', None)
        if imgs is None or self.img_backbone is None or \
                self.refine_head is None or not self.refine_head.use_image:
            return None, None, None
        return image_branch(self.img_backbone, self.img_neck, imgs)

    def _stage1(self, batch_inputs_dict, batch_data_samples, metas):
        x = self.extract_feat(batch_inputs_dict, metas)
        preds, targets = self.backbone(x, batch_data_samples)
        bev, self._bev = self._bev, None
        return self.bbox_head(preds), targets, bev

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs) -> dict:
        metas = [s.metainfo for s in batch_data_samples]
        img_feats, bright, img_hw = self._images(batch_inputs_dict)
        preds, targets, bev = self._stage1(batch_inputs_dict,
                                           batch_data_samples, metas)
        losses = self.bbox_head.loss(preds, targets, metas=metas)
        if self.refine_head is not None:
            with torch.no_grad():
                # predict() permutes the tensors of its dicts in place
                det = [{k: v.detach() if torch.is_tensor(v) else v
                        for k, v in p.items()} for p in preds]
                dets = self.bbox_head.predict(det, metas)
            gts = [s.gt_instances_3d for s in batch_data_samples]
            losses.update(refine_losses(self.refine_head, dets, gts, bev,
                                        img_feats, metas, bright, img_hw))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        img_feats, bright, img_hw = self._images(batch_inputs_dict)
        preds, _, bev = self._stage1(batch_inputs_dict, batch_data_samples,
                                     metas)
        results = self.bbox_head.predict(preds, metas)
        if self.refine_head is not None:
            results = refine_predictions(self.refine_head, results, bev,
                                         img_feats, metas, bright, img_hw)
        return self.add_pred_to_datasample(batch_data_samples, results)
