"""CenterPoint with evidence anchors and the physical sea surface.

Evidence anchors. LiDAR sees only the sensor-facing side of a hull: 72% of
the GT centres have no return in their 0.8 m BEV cell, and CenterPoint's
centre error doubles there (val, 1.40 vs 0.59 m median; 2.73 m for hulls
>30 m). The centroid of the box's own returns holds evidence (a return in
its cell for 53-99% of boxes, vs 1-6% for the nearest corner and 12-24% for
the centre; progress/20260927_231500_evidence-anchor-analysis), so the
heatmap peaks there and an anchor-to-centre vector is regressed.

Sea surface. :class:`SeaSurface` (IMU tilt, Kalman mean level, GP waves)
fuses every hull's waterline with its neighbours' and the mean level by
their learned uncertainties; in training this supervises the heads (the
final configs use it as training supervision only, ``sea_at_test=False``).
"""
from typing import List

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet.models.utils import multi_apply  # noqa: F401
from torch import Tensor

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.models.dense_heads.centerpoint_head import CenterHead
from mmdet3d.models.task_modules.coders.centerpoint_bbox_coders import \
    CenterPointBBoxCoder
from mmdet3d.models.utils import (clip_sigmoid, draw_heatmap_gaussian,
                                  gaussian_radius)
from mmdet3d.registry import MODELS, TASK_UTILS, TRANSFORMS
from .sea_surface import LOG_SIG_RANGE, SeaSurface

LOG_DIM_RANGE = (-5.0, 6.0)


@TRANSFORMS.register_module()
class EvidenceAnchor(BaseTransform):
    """BEV anchor of every GT box: the centroid of its own LiDAR returns.

    Place after the augmentation and the last box filter, so the anchors are
    computed from the points and boxes the network sees. Adds
    ``gt_anchors_3d`` (float32 [N, 2]); pack it with
    :class:`Pack3DDetInputsAnchor`. Boxes without a return keep their centre.

    Args:
        mode (str): 'centroid' (evidence anchor) or 'centre' (plain
            CenterPoint; the anchor-to-centre vector is then zero).
        margin (float): Tolerance around the box in x / y (m).
        z_margin (float): Tolerance below the bottom / above the top (m).
    """

    def __init__(self, mode: str = 'centroid', margin: float = 0.3,
                 z_margin: float = 0.5) -> None:
        assert mode in ('centroid', 'centre')
        self.mode = mode
        self.margin = float(margin)
        self.z_margin = float(z_margin)

    def transform(self, results: dict) -> dict:
        t = results['gt_bboxes_3d'].tensor.numpy().astype(np.float64)
        anchors = t[:, :2].copy()
        if self.mode == 'centroid' and len(t):
            pts = results['points'].tensor.numpy()[:, :3].astype(np.float64)
            for i, (x, y, zb, l, w, h, yaw) in enumerate(t[:, :7]):
                c, s = np.cos(yaw), np.sin(yaw)
                dx, dy = pts[:, 0] - x, pts[:, 1] - y
                inside = (np.abs(dx * c + dy * s) <= l / 2 + self.margin) & \
                    (np.abs(-dx * s + dy * c) <= w / 2 + self.margin) & \
                    (pts[:, 2] >= zb - self.z_margin) & \
                    (pts[:, 2] <= zb + h + self.z_margin)
                if inside.any():
                    anchors[i] = pts[inside, :2].mean(0)
        results['gt_anchors_3d'] = torch.from_numpy(
            anchors.astype(np.float32))
        return results

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(mode={self.mode!r})'


@TRANSFORMS.register_module()
class Pack3DDetInputsAnchor(Pack3DDetInputs):
    """Pack3DDetInputs that also packs ``gt_anchors_3d`` into
    ``gt_instances_3d.anchors_3d``."""
    INSTANCEDATA_3D_KEYS = Pack3DDetInputs.INSTANCEDATA_3D_KEYS + [
        'gt_anchors_3d'
    ]


def a2c_offset(code: Tensor, dim: Tensor, yaw: Tensor,
               frame: str) -> Tensor:
    """Anchor-to-centre vector (m, [..., 2]) from its code.

    'bev': the code is the vector in metres. 'local': the code is the
    centre's position in the box frame as fractions of the hull, (alpha
    along the length, beta across the width); dim (m) and yaw are the box's.
    """
    if frame == 'bev':
        return code
    yaw = yaw.reshape(code.shape[:-1])
    c, s = torch.cos(yaw), torch.sin(yaw)
    dl = code[..., 0] * dim[..., 0]
    dw = code[..., 1] * dim[..., 1]
    return torch.stack([dl * c - dw * s, dl * s + dw * c], dim=-1)


def a2c_code(vec: Tensor, dim: Tensor, yaw: Tensor, frame: str) -> Tensor:
    """Inverse of :func:`a2c_offset` (vec: anchor -> centre, m)."""
    if frame == 'bev':
        return vec
    c, s = torch.cos(yaw), torch.sin(yaw)
    al = (vec[..., 0] * c + vec[..., 1] * s) / dim[..., 0]
    be = (-vec[..., 0] * s + vec[..., 1] * c) / dim[..., 1]
    return torch.stack([al, be], dim=-1)


@TASK_UTILS.register_module()
class EvidenceCenterPointBBoxCoder(CenterPointBBoxCoder):
    """Decodes anchor + anchor-to-centre vector into the box centre.

    The 'vel' input carries [a2c (2), log_b (1), fb (1)] and the decoded box
    is [x, y, z, l, w, h, yaw, log_b, fb] (``code_size=9``); the head turns
    it into a 7-dof box after refining the heights on the sea surface.
    """

    def __init__(self, *args, a2c_scale: float = 2.0,
                 a2c_frame: str = 'bev', **kwargs) -> None:
        super().__init__(*args, **kwargs)
        assert a2c_frame in ('bev', 'local')
        self.a2c_scale = float(a2c_scale)
        self.a2c_frame = a2c_frame

    def decode(self, heat, rot_sine, rot_cosine, hei, dim, vel, reg=None,
               task_id=-1) -> List[dict]:
        batch = heat.shape[0]
        K = self.max_num
        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        def gather(x, c):
            return self._transpose_and_gather_feat(x, inds).view(batch, K, c)

        xs, ys = xs.view(batch, K, 1), ys.view(batch, K, 1)
        if reg is not None:
            reg = gather(reg, 2)
            xs, ys = xs + reg[..., 0:1], ys + reg[..., 1:2]
        else:
            xs, ys = xs + 0.5, ys + 0.5
        extra = gather(vel, 4)
        cell_x = self.out_size_factor * self.voxel_size[0]
        cell_y = self.out_size_factor * self.voxel_size[1]
        rot = torch.atan2(gather(rot_sine, 1), gather(rot_cosine, 1))
        dim = gather(dim, 3)
        off = a2c_offset(extra[..., 0:2] * self.a2c_scale, dim, rot,
                         self.a2c_frame)
        xs = xs * cell_x + self.pc_range[0] + off[..., 0:1]
        ys = ys * cell_y + self.pc_range[1] + off[..., 1:2]
        boxes = torch.cat(
            [xs, ys, gather(hei, 1), dim, rot, extra[..., 2:4]], dim=2)
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)
        pcr = torch.as_tensor(self.post_center_range, device=heat.device)
        mask = (boxes[..., :3] >= pcr[:3]).all(2)
        mask &= (boxes[..., :3] <= pcr[3:]).all(2)
        if self.score_threshold:
            mask &= scores > self.score_threshold
        return [
            dict(bboxes=boxes[i, mask[i]], scores=scores[i, mask[i]],
                 labels=clses[i, mask[i]]) for i in range(batch)
        ]


@MODELS.register_module()
class EvidenceSeaCenterHead(CenterHead):
    """CenterHead with evidence anchors and (optionally) the sea surface.

    Heads: ``a2c`` (2, anchor-to-centre vector / ``a2c_scale``) always;
    ``sig`` (1, log Laplace scale of the own waterline) and ``fb`` (1,
    freeboard: final bottom - sea) with ``sea_surface``. Regression code
    (``code_weights``, 10): anchor sub-cell offset (2), gravity-centre z
    (1), log l, w, h (3), sin / cos yaw (2), a2c (2). Needs the
    ``gt_anchors_3d`` of :class:`EvidenceAnchor` in training and, with the
    sea surface, the ``sea_up`` / ``sea_up_valid`` / ``sea_up_rot`` metas of
    ``LoadSeaUp``.

    Training: the observations of the sea surface are the predictions at
    the GT anchor cells (teacher forcing), with positions, sizes and own
    heights detached; the fused height is supervised by an L1 on the GT
    gravity-centre z, the waterline scales by a Laplace NLL and the
    posterior mean level by an L1 on the GT level. Test: with
    ``sea_at_test=False`` the boxes keep the head's own heights; with True
    the most confident detections after NMS observe the sea, every
    detection's bottom becomes sea + fb and the mean level is streamed over
    each sequence (evaluate with SequentialChunkSampler).

    Args:
        a2c_scale (float): Metres per unit of the a2c code ('bev'), or
            hull fractions per unit ('local').
        a2c_frame (str): 'bev' (vector in metres) or 'local' (centre as
            fractions of the box length / width in the box frame: bounded,
            dimensionless, the same range for a 6 m boat and a 70 m ship).
        sea_surface (dict, optional): :class:`SeaSurface` arguments; None
            gives plain evidence-anchor CenterPoint.
        loss_final_height_weight (float): L1 on the fused height.
        loss_sigma_weight (float): Laplace NLL of the waterline scales.
        loss_level_weight (float): L1 of the posterior mean level.
        sea_at_test (bool): Put the detections on the sea surface at test
            time. False keeps the head's own heights (the sea surface then
            only acts through the training losses).
    """

    def __init__(self,
                 *args,
                 a2c_scale: float = 2.0,
                 a2c_frame: str = 'bev',
                 sea_surface: dict = None,
                 loss_final_height_weight: float = 0.25,
                 loss_sigma_weight: float = 0.25,
                 loss_level_weight: float = 0.5,
                 sea_at_test: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        heads = self.task_heads[0].heads
        assert 'a2c' in heads, "needs common_heads['a2c'] = (2, 2)"
        assert a2c_frame in ('bev', 'local')
        self.a2c_scale = float(a2c_scale)
        self.a2c_frame = a2c_frame
        self.sea = None
        if sea_surface is not None:
            assert 'sig' in heads and 'fb' in heads, \
                "the sea surface needs common_heads 'sig' and 'fb'"
            self.sea = SeaSurface(**sea_surface)
        self.loss_final_height_weight = float(loss_final_height_weight)
        self.loss_sigma_weight = float(loss_sigma_weight)
        self.loss_level_weight = float(loss_level_weight)
        self.sea_at_test = bool(sea_at_test)
        self._metas = None

    # ------------------------------------------------------------ targets
    def get_targets_single(self, gt_instances_3d):
        """CenterHead targets with the heatmap peak at the evidence anchor.

        Returns per task: heatmap [n_cls, H, W], code [max_objs, 10],
        anchor-cell index [max_objs] and mask [max_objs].
        """
        labels = gt_instances_3d.labels_3d
        device = labels.device
        b3d = gt_instances_3d.bboxes_3d
        boxes = torch.cat((b3d.gravity_center, b3d.tensor[:, 3:7]),
                          dim=1).to(device)
        if 'anchors_3d' in gt_instances_3d:
            anchors = gt_instances_3d.anchors_3d.to(device).float()
        else:
            anchors = boxes[:, :2]
        cfg = self.train_cfg
        max_objs = cfg['max_objs'] * cfg['dense_reg']
        pcr, vs, osf = cfg['point_cloud_range'], cfg['voxel_size'], \
            cfg['out_size_factor']
        fw = int(cfg['grid_size'][0]) // osf
        fh = int(cfg['grid_size'][1]) // osf
        heatmaps, codes, inds, masks = [], [], [], []
        flag = 0
        for names in self.class_names:
            heatmap = boxes.new_zeros(len(names), fh, fw)
            code = boxes.new_zeros(max_objs, 10)
            ind = labels.new_zeros(max_objs, dtype=torch.int64)
            mask = boxes.new_zeros(max_objs, dtype=torch.uint8)
            sel = torch.cat([
                torch.where(labels == flag + j)[0] for j in range(len(names))
            ])
            cls_ids = labels[sel] - flag
            flag += len(names)
            for k in range(min(len(sel), max_objs)):
                b, a = boxes[sel[k]], anchors[sel[k]]
                length = b[3] / vs[0] / osf
                width = b[4] / vs[1] / osf
                if not (width > 0 and length > 0):
                    continue
                radius = gaussian_radius(
                    (width, length), min_overlap=cfg['gaussian_overlap'])
                radius = max(cfg['min_radius'], int(radius))
                axy = torch.stack([(a[0] - pcr[0]) / vs[0] / osf,
                                   (a[1] - pcr[1]) / vs[1] / osf])
                aint = axy.to(torch.int32)
                if not (0 <= aint[0] < fw and 0 <= aint[1] < fh):
                    continue
                draw_heatmap_gaussian(heatmap[cls_ids[k]], aint, radius)
                ind[k] = aint[1] * fw + aint[0]
                mask[k] = 1
                dim = b[3:6].log() if self.norm_bbox else b[3:6]
                code[k] = torch.cat([
                    axy - aint.float(), b[2:3], dim,
                    torch.sin(b[6:7]), torch.cos(b[6:7]),
                    a2c_code(b[:2] - a, b[3:5], b[6], self.a2c_frame) /
                    self.a2c_scale
                ])
            heatmaps.append(heatmap)
            codes.append(code)
            inds.append(ind)
            masks.append(mask)
        return heatmaps, codes, inds, masks

    # ------------------------------------------------------------- losses
    def loss(self, pts_feats, batch_data_samples, *args, **kwargs):
        """Keeps the metas at hand: the sea surface needs the IMU tilt."""
        self._metas = [s.metainfo for s in batch_data_samples]
        try:
            return super().loss(pts_feats, batch_data_samples, *args,
                                **kwargs)
        finally:
            self._metas = None

    def _gather(self, x: Tensor, ind: Tensor) -> Tensor:
        """[B, C, H, W] map at [B, K] flat indices -> [B, K, C]."""
        x = x.permute(0, 2, 3, 1).contiguous()
        return self._gather_feat(x.view(x.size(0), -1, x.size(3)), ind)

    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args,
                     **kwargs):
        heatmaps, codes, inds, masks = self.get_targets(batch_gt_instances_3d)
        loss_dict = dict()
        at_gt = []
        for task_id, preds_dict in enumerate(preds_dicts):
            p = preds_dict[0]
            p['heatmap'] = clip_sigmoid(p['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            loss_dict[f'task{task_id}.loss_heatmap'] = self.loss_cls(
                p['heatmap'], heatmaps[task_id], avg_factor=max(num_pos, 1))
            target = codes[task_id]
            pred = self._gather(
                torch.cat([p[k] for k in ('reg', 'height', 'dim', 'rot',
                                          'a2c')], dim=1), inds[task_id])
            num = masks[task_id].float().sum()
            mask = masks[task_id].unsqueeze(2).expand_as(target).float()
            mask *= (~torch.isnan(target)).float()
            weights = mask * mask.new_tensor(self.train_cfg['code_weights'])
            loss_dict[f'task{task_id}.loss_bbox'] = self.loss_bbox(
                pred, target, weights, avg_factor=(num + 1e-4))
            if self.sea is not None:
                score = self._gather(p['heatmap'].detach(),
                                     inds[task_id]).max(-1).values
                at_gt.append(dict(
                    pred=pred, tgt=target, ind=inds[task_id],
                    mask=masks[task_id].float(), score=score,
                    log_b=self._gather(p['sig'], inds[task_id])[..., 0],
                    fb=self._gather(p['fb'], inds[task_id])[..., 0]))
        if self.sea is not None:
            with torch.autocast('cuda', enabled=False):
                loss_dict.update(self._sea_losses(at_gt))
        return loss_dict

    def _cell_to_m(self, ind: Tensor, code: Tensor):
        """Box centre (m) from the anchor cell index and the box code
        [sub-cell offset (2), z, log l w h (3), sin, cos, a2c (2)]."""
        cfg = self.train_cfg
        osf, vs, pcr = cfg['out_size_factor'], cfg['voxel_size'], \
            cfg['point_cloud_range']
        fw = int(cfg['grid_size'][0]) // osf
        cx = (ind % fw).float() + code[..., 0]
        cy = (ind // fw).float() + code[..., 1]
        dim = code[..., 3:6].clamp(*LOG_DIM_RANGE).exp() if self.norm_bbox \
            else code[..., 3:6]
        yaw = torch.atan2(code[..., 6], code[..., 7])
        off = a2c_offset(code[..., 8:10] * self.a2c_scale, dim, yaw,
                         self.a2c_frame)
        x = cx * vs[0] * osf + pcr[0] + off[..., 0]
        y = cy * vs[1] * osf + pcr[1] + off[..., 1]
        return x, y

    def _sea_losses(self, at_gt: List[dict]) -> dict:
        """Sea surface on the predictions at the GT anchor cells."""
        cat = {k: torch.cat([g[k] for g in at_gt], dim=1) for k in at_gt[0]}
        valid = cat['mask'] > 0
        # compact: the GT objects of every sample first, padded to n >= 1
        n = max(1, int(valid.sum(1).max()))
        order = torch.sort(valid.float(), dim=1, descending=True,
                           stable=True).indices[:, :n]
        g = {}
        for k, v in cat.items():
            idx = order if v.dim() == 2 else \
                order[..., None].expand(-1, -1, v.shape[-1])
            g[k] = v.gather(1, idx)
        m = g['mask']
        pred, tgt = g['pred'].float(), g['tgt']
        logd = pred[..., 3:6].clamp(*LOG_DIM_RANGE).detach()
        L, h = logd[..., 0].exp(), logd[..., 2].exp()
        x, y = self._cell_to_m(g['ind'], pred.detach())
        bottom = pred[..., 2].detach() - 0.5 * h
        z_gt = tgt[..., 2]
        bottom_gt = z_gt - 0.5 * tgt[..., 5].exp()
        gx, gy = self._cell_to_m(g['ind'], tgt)
        metas = self._metas or [{} for _ in range(m.shape[0])]
        tilt = self.sea.tilt(metas, m.device)
        levels = []
        for i in range(m.shape[0]):
            v = m[i] > 0
            levels.append(self.sea.gt_level(
                torch.stack([gx[i][v], gy[i][v], bottom_gt[i][v]], dim=1),
                tilt[i]))
        c_prior, P_prior = self.sea.level_prior(metas, levels)
        sea, c_post, _ = self.sea.solve(
            tilt, c_prior, P_prior,
            obs=dict(x=x, y=y, L=L, bottom=bottom, log_b=g['log_b'].float(),
                     score=g['score'].float(), mask=m),
            qry=dict(x=x, y=y, L=L))
        z = sea + 0.5 * h + g['fb'].float()
        denom = m.sum().clamp_min(1.0)
        losses = dict(
            loss_sea_height=self.loss_final_height_weight *
            ((z - z_gt).abs() * m).sum() / denom)
        err = (bottom - bottom_gt).abs().detach()
        log_b = g['log_b'].float().clamp(*LOG_SIG_RANGE)
        losses['loss_sigma'] = self.loss_sigma_weight * (
            (err * (-log_b).exp() + log_b) * m).sum() / denom
        lv = [(c_post[i] - gt).abs() for i, gt in enumerate(levels)
              if gt is not None]
        losses['loss_level'] = self.loss_level_weight * (
            torch.stack(lv).mean() if lv else 0 * c_post.sum())
        return losses

    # ---------------------------------------------------------- inference
    def predict_by_feat(self, preds_dicts, batch_input_metas, *args,
                        **kwargs):
        for preds_dict in preds_dicts:
            p = preds_dict[0]
            extra = [p['sig'], p['fb']] if self.sea is not None else \
                [torch.zeros_like(p['a2c'])]
            # decoded by EvidenceCenterPointBBoxCoder from the 'vel' slot
            p['vel'] = torch.cat([p['a2c']] + extra, dim=1)
        results = super().predict_by_feat(preds_dicts, batch_input_metas,
                                          *args, **kwargs)
        for meta, r in zip(batch_input_metas, results):
            b = r.bboxes_3d.tensor  # z is the bottom, cols 7/8: log_b, fb
            box = b[:, :7].clone()
            if self.sea is not None and self.sea_at_test:
                box[:, 2] = self._sea_bottoms(meta, b, r.scores_3d)
            r.bboxes_3d = meta['box_type_3d'](box, box_dim=7)
        return results

    def _sea_bottoms(self, meta: dict, b: Tensor, scores: Tensor) -> Tensor:
        """Bottoms of one frame's detections on the sea surface."""
        with torch.autocast('cuda', enabled=False):
            b = b.float()
            tilt = self.sea.tilt([meta], b.device)
            c_prior, P_prior = self.sea.level_prior([meta])
            if b.shape[0] == 0:
                if not self.training:
                    self.sea.update_stream([meta], c_prior, P_prior)
                return b[:, 2]
            x, y, L = b[None, :, 0], b[None, :, 1], b[None, :, 3]
            sea, c_post, P_post = self.sea.solve(
                tilt, c_prior, P_prior,
                obs=dict(x=x, y=y, L=L, bottom=b[None, :, 2],
                         log_b=b[None, :, 7], score=scores[None].float(),
                         mask=torch.ones_like(x)),
                qry=dict(x=x, y=y, L=L))
            if not self.training:  # stage-1 proposals of MariFusion
                self.sea.update_stream([meta], c_post, P_post)
            return sea[0] + b[:, 8]
