"""MariFusion: two-stage sparse instance fusion of LiDAR and the two front
cameras on top of CenterPoint.

Stage 1 is CenterPoint (LiDAR only). Stage 2 refines every detection with
image evidence, sampled only where the detection projects:

* image sampling -- each detection's 3D box is projected into both front
  cameras (full-resolution horizon band); a G x G grid over its projection,
  widened sideways and upwards for box errors but never extended below the
  projected waterline (the bottom face), samples two FPN levels. Water
  mirrors hulls and lights directly below the waterline, so that region is
  masked out (reflection-aware sampling);
* per-camera reliability gate -- g_v = sigmoid(MLP(valid fraction, image
  brightness, range)), zero when nothing projects into camera v; modality
  dropout in training (all cameras p_all, one camera p_single). Outside the
  cameras (76% of the objects) the refinement is LiDAR-only by construction;
* ray-split refinement -- the residual along the viewing ray (range) and in
  height come from the LiDAR BEV feature alone; the residual across the ray
  (bearing), the size, the heading and the score come from the fused
  feature. A camera pins angles (1 px ~ 5.5 cm across at 100 m) but not
  range (1 px of waterline ~ 1.8 m along it); LiDAR is the opposite.

``use_image=False`` gives the same second stage without cameras (the
control that separates the camera's contribution from the refinement's).
"""
import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import box_iou_rotated
from mmengine.model import BaseModule
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.detectors import CenterPoint
from mmdet3d.registry import MODELS


def _mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.ReLU(inplace=True)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def _corners(b: Tensor) -> Tensor:
    """[n, 7] boxes (x, y, z_bottom, l, w, h, yaw) -> [n, 8, 3] corners;
    0-3 bottom, 4-7 top."""
    x, y, z, l, w, h, yaw = b.unbind(-1)
    c, s = torch.cos(yaw), torch.sin(yaw)
    du = torch.stack([l, l, -l, -l], -1) / 2
    dv = torch.stack([w, -w, -w, w], -1) / 2
    cx = x[:, None] + du * c[:, None] - dv * s[:, None]
    cy = y[:, None] + du * s[:, None] + dv * c[:, None]
    bot = torch.stack([cx, cy, z[:, None].expand_as(cx)], -1)
    top = torch.stack([cx, cy, (z + h)[:, None].expand_as(cx)], -1)
    return torch.cat([bot, top], 1)


@MODELS.register_module()
class RaySplitFusionRefineHead(BaseModule):
    """Second stage of MariFusion (see module docstring).

    Args:
        num_classes (int): Detection classes.
        bev_channels (int): Channels of the LiDAR BEV feature.
        img_channels (int): Channels of each image FPN level.
        hidden (int): Width of the instance features.
        pc_range (list): x_min, y_min, ..., x_max, y_max of the BEV map.
        use_image (bool): False: the same stage without cameras.
        num_levels (int): Image FPN levels sampled (strides 8, 16, ...).
        grid (int): G, the G x G sampling grid per camera.
        widen (float): Horizontal / upward margin of the grid, as a
            fraction of the projected box size.
        waterline_margin (float): Rows (px) allowed below the waterline.
        drop_all, drop_single (float): Training modality dropout.
        pos_iou (float): BEV IoU for a proposal to be regressed.
        max_train_props (int): Stage-1 proposals per frame in training.
        gt_jitter (bool): Add one jittered copy of every GT box as a
            proposal in training.
        score_alpha (float): final = s1^(1 - a) * sigmoid(q)^a.
        range_scale, cross_scale (float): Metres per unit of the residuals
            along / across the ray.
    """

    def __init__(self,
                 num_classes: int = 5,
                 bev_channels: int = 512,
                 img_channels: int = 256,
                 hidden: int = 256,
                 pc_range=(-160.0, -160.0, -8.0, 160.0, 160.0, 24.0),
                 use_image: bool = True,
                 num_levels: int = 2,
                 grid: int = 4,
                 widen: float = 0.2,
                 waterline_margin: float = 2.0,
                 drop_all: float = 0.25,
                 drop_single: float = 0.1,
                 pos_iou: float = 0.25,
                 max_train_props: int = 128,
                 gt_jitter: bool = True,
                 score_alpha: float = 0.5,
                 range_scale: float = 2.0,
                 cross_scale: float = 1.0,
                 loss_reg_weight: float = 1.0,
                 loss_q_weight: float = 1.0,
                 init_cfg=None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.pc_range = [float(v) for v in pc_range]
        self.use_image = use_image
        self.num_levels = num_levels
        self.grid = grid
        self.widen = widen
        self.wl_margin = waterline_margin
        self.drop_all, self.drop_single = drop_all, drop_single
        self.pos_iou = pos_iou
        self.max_train_props = max_train_props
        self.gt_jitter = gt_jitter
        self.alpha = score_alpha
        self.range_scale, self.cross_scale = range_scale, cross_scale
        self.w_reg, self.w_q = loss_reg_weight, loss_q_weight
        # LiDAR instance feature: centre + 4 BEV corners of the box
        self.bev_proj = _mlp(5 * bev_channels, hidden, hidden)
        self.box_enc = _mlp(10 + num_classes, hidden, hidden)
        if use_image:
            self.img_proj = nn.Linear(num_levels * img_channels, hidden)
            self.img_pos = _mlp(2, hidden // 2, hidden)
            self.q_proj = nn.Linear(hidden, hidden)
            self.img_out = nn.Linear(hidden, hidden)
            self.gate = _mlp(5, 64, 1)
        self.lidar_reg = _mlp(hidden, hidden, 2)       # d_range, d_z
        self.fused_reg = _mlp(hidden, hidden, 5)       # d_cross, dlog lwh, d_yaw
        self.quality = _mlp(hidden, hidden, 1)
        self._last_gate = None

    # ---------------------------------------------------------------- utils
    def _ray(self, b):
        r = torch.hypot(b[:, 0], b[:, 1]).clamp_min(1e-3)
        u = torch.stack([b[:, 0], b[:, 1]], -1) / r[:, None]
        t = torch.stack([-u[:, 1], u[:, 0]], -1)
        return r, u, t

    def _bev_feat(self, bev: Tensor, b: Tensor) -> Tensor:
        """[C, H, W] map, [n, 7] boxes -> [n, 5 C] (centre + BEV corners)."""
        pts = torch.cat([b[:, None, :2], _corners(b)[:, :4, :2]], 1)
        x0, y0, _, x1, y1, _ = self.pc_range
        g = torch.stack([(pts[..., 0] - x0) / (x1 - x0) * 2 - 1,
                         (pts[..., 1] - y0) / (y1 - y0) * 2 - 1], -1)
        f = F.grid_sample(bev[None].float(), g[None], align_corners=False)
        return f[0].permute(1, 2, 0).reshape(b.shape[0], -1)

    def _box_enc(self, b, s, lab):
        r, u, _ = self._ray(b)
        theta = torch.atan2(u[:, 1], u[:, 0])
        rel = b[:, 6] - theta
        e = torch.stack([
            r / 160.0, torch.sin(theta), torch.cos(theta), b[:, 2] / 8.0,
            b[:, 3].clamp_min(1e-2).log(), b[:, 4].clamp_min(1e-2).log(),
            b[:, 5].clamp_min(1e-2).log(), torch.sin(rel), torch.cos(rel), s
        ], -1)
        return torch.cat([e, F.one_hot(lab.long(), self.num_classes).float()],
                         -1)

    def _project(self, pts, meta, v):
        """[m, 3] augmented-LiDAR points -> pixel (u, v) [m, 2], depth [m]."""
        dev = pts.device
        l2i = torch.as_tensor(meta['lidar2img'][v], dtype=torch.float32,
                              device=dev)
        aug = torch.as_tensor(meta.get('lidar_aug_matrix', torch.eye(4)),
                              dtype=torch.float32, device=dev)
        ia = torch.as_tensor(meta['img_aug_matrix'][v], dtype=torch.float32,
                             device=dev)
        hom = torch.cat([pts, torch.ones_like(pts[:, :1])], -1)
        q = (l2i @ torch.linalg.inv(aug) @ hom.T).T
        d = q[:, 2]
        uv = q[:, :2] / d.clamp_min(1e-3)[:, None]
        uv = uv @ ia[:2, :2].T + ia[:2, 3]
        return uv, d

    def _img_feat(self, feats, meta, b, q, bright, img_hw):
        """Masked attention over the reflection-aware grid, per camera.

        feats: list over levels of [N_cam, C, h, w]. Returns image features
        [N_cam, n, hidden] and the valid fraction [N_cam, n].
        """
        n, G = b.shape[0], self.grid
        H, W = img_hw
        out, frac = [], []
        cor = _corners(b)                                   # [n, 8, 3]
        for v in range(feats[0].shape[0]):
            uv, d = self._project(cor.reshape(-1, 3), meta, v)
            uv, d = uv.view(n, 8, 2), d.view(n, 8)
            ok_c = d > 0.5
            big = torch.full_like(uv[..., 0], 1e6)
            umin = torch.where(ok_c, uv[..., 0], big).min(1).values
            umax = torch.where(ok_c, uv[..., 0], -big).max(1).values
            vmin = torch.where(ok_c, uv[..., 1], big).min(1).values
            # waterline: the lowest projected bottom corner
            vwl = torch.where(ok_c[:, :4], uv[:, :4, 1], -big[:, :4]).max(1) \
                .values
            du = (umax - umin) * self.widen
            u0, u1 = umin - du, umax + du
            v0 = vmin - (vwl - vmin) * self.widen
            v1 = vwl + self.wl_margin
            lin = (torch.arange(G, device=b.device).float() + 0.5) / G
            gu = u0[:, None] + (u1 - u0)[:, None] * lin[None]    # [n, G]
            gv = v0[:, None] + (v1 - v0)[:, None] * lin[None]
            gu = gu[:, None, :].expand(n, G, G).reshape(n, G * G)
            gv = gv[:, :, None].expand(n, G, G).reshape(n, G * G)
            valid = (ok_c.sum(1) >= 4)[:, None] & (gu >= 0) & (gu < W) & \
                (gv >= 0) & (gv < H)
            # masked samples must not carry NaN / inf into the attention
            gu = torch.nan_to_num(gu).clamp(-W, 2 * W)
            gv = torch.nan_to_num(gv).clamp(-H, 2 * H)
            u0 = torch.nan_to_num(u0).clamp(-W, 2 * W)
            u1 = torch.nan_to_num(u1).clamp(-W, 2 * W)
            v0 = torch.nan_to_num(v0).clamp(-H, 2 * H)
            v1 = torch.nan_to_num(v1).clamp(-H, 2 * H)
            grid = torch.stack([gu / W * 2 - 1, gv / H * 2 - 1], -1)
            smp = [F.grid_sample(f[v:v + 1].float(), grid[None],
                                 align_corners=False)[0].permute(1, 2, 0)
                   for f in feats[:self.num_levels]]           # [n, G2, C]
            k = self.img_proj(torch.cat(smp, -1))
            pos = torch.stack([(gu - u0[:, None]) /
                               (u1 - u0).clamp_min(1)[:, None],
                               (gv - v0[:, None]) /
                               (v1 - v0).clamp_min(1)[:, None]], -1)
            k = (k + self.img_pos(pos.clamp(0, 1))) * valid[..., None]
            att = (self.q_proj(q)[:, None, :] * k).sum(-1) / math.sqrt(
                k.shape[-1])
            att = att.masked_fill(~valid, -1e4).softmax(-1) * valid
            out.append(self.img_out((att[..., None] * k).sum(1)))
            frac.append(valid.float().mean(1))
        return torch.stack(out), torch.stack(frac)

    # -------------------------------------------------------------- forward
    def forward(self, props: List[Tensor], scores: List[Tensor],
                labels: List[Tensor], bev: Tensor,
                img_feats: Optional[List[Tensor]], metas: List[dict],
                bright: Optional[Tensor], img_hw=None) -> List[dict]:
        """Per frame: residuals, quality logit and gates of its proposals."""
        outs = []
        for i, (b, s, lab) in enumerate(zip(props, scores, labels)):
            if b.shape[0] == 0:
                outs.append(None)
                continue
            b = self.sanitize(b.float())
            f_bev = self.bev_proj(self._bev_feat(bev[i], b)) + \
                self.box_enc(self._box_enc(b, s.float(), lab))
            fused = f_bev
            gates = None
            if self.use_image and img_feats is not None:
                n_cam = len(metas[i]['lidar2img'])
                lv = [f[i * n_cam:(i + 1) * n_cam] for f in img_feats]
                f_img, frac = self._img_feat(lv, metas[i], b, f_bev,
                                             bright[i], img_hw)
                r = torch.hypot(b[:, 0], b[:, 1]) / 160.0
                gin = torch.stack([
                    frac, bright[i][:, None, 0].expand_as(frac),
                    bright[i][:, None, 1].expand_as(frac),
                    r[None].expand_as(frac), (frac > 0).float()
                ], -1)
                gates = torch.sigmoid(self.gate(gin))[..., 0] * (frac > 0)
                if self.training:
                    keep = torch.ones_like(gates[:, :1])
                    if torch.rand(()) < self.drop_all:
                        keep = keep * 0
                    single = (torch.rand(gates.shape[0], 1,
                                         device=gates.device) >=
                              self.drop_single).float()
                    gates = gates * keep * single
                fused = f_bev + (gates[..., None] * f_img).sum(0)
            lr = self.lidar_reg(f_bev)
            fr = self.fused_reg(fused)
            outs.append(dict(d_range=lr[:, 0], d_z=lr[:, 1],
                             d_cross=fr[:, 0], d_dim=fr[:, 1:4],
                             d_yaw=fr[:, 4], q=self.quality(fused)[:, 0],
                             gates=gates))
        return outs

    @staticmethod
    def sanitize(b: Tensor) -> Tensor:
        """Finite boxes with sizes in [0.1, 200] m (stage-1 decodes of an
        untrained model can overflow)."""
        b = torch.nan_to_num(b, nan=0.0, posinf=1e3, neginf=-1e3)
        return torch.cat([b[:, :3].clamp(-1e3, 1e3),
                          b[:, 3:6].clamp(0.1, 200.0), b[:, 6:7]], -1)

    def refine_boxes(self, b: Tensor, o: dict) -> Tensor:
        """Refined [n, 7] boxes (bottom z)."""
        b = self.sanitize(b)
        _, u, t = self._ray(b)
        c = b[:, :2] + (o['d_range'] * self.range_scale)[:, None] * u + \
            (o['d_cross'] * self.cross_scale)[:, None] * t
        dims = b[:, 3:6] * o['d_dim'].clamp(-1, 1).exp()
        return torch.cat([c, (b[:, 2] + o['d_z'])[:, None], dims,
                          (b[:, 6] + o['d_yaw'])[:, None]], -1)

    def score(self, s1: Tensor, o: dict) -> Tensor:
        return s1.clamp_min(1e-6)**(1 - self.alpha) * \
            torch.sigmoid(o['q'])**self.alpha

    # ---------------------------------------------------------------- train
    def sample_props(self, dets: List[InstanceData],
                     gts: List[InstanceData]):
        """Stage-1 detections (top-K) plus one jittered copy of each GT."""
        props, scores, labels = [], [], []
        for d, g in zip(dets, gts):
            b = d.bboxes_3d.tensor[:, :7].detach()
            s = d.scores_3d.detach()
            lab = d.labels_3d.detach()
            if b.shape[0] > self.max_train_props:
                k = s.topk(self.max_train_props).indices
                b, s, lab = b[k], s[k], lab[k]
            gb = g.bboxes_3d.tensor[:, :7].to(b.device).float()
            if self.gt_jitter and gb.shape[0]:
                _, u, t = self._ray(gb)
                sig = (gb[:, 3:5].max(1).values / 10).clamp(0.3, 3.0)
                j = gb.clone()
                j[:, :2] += (torch.randn_like(sig) * sig)[:, None] * u + \
                    (torch.randn_like(sig) * sig * 0.5)[:, None] * t
                j[:, 2] += torch.randn_like(sig) * 0.3
                j[:, 3:6] *= (torch.randn_like(j[:, 3:6]) * 0.1).exp()
                j[:, 6] += torch.randn_like(sig) * 0.1
                b = torch.cat([b, j])
                s = torch.cat([s, torch.rand_like(sig) * 0.8 + 0.1])
                lab = torch.cat([lab, g.labels_3d.to(lab.device).long()])
            props.append(b)
            scores.append(s)
            labels.append(lab)
        return props, scores, labels

    def loss(self, outs, props, labels, gts) -> dict:
        reg, qls, n_pos = [], [], 0
        for o, b, lab, g in zip(outs, props, labels, gts):
            if o is None:
                continue
            b = self.sanitize(b.float())
            gb = g.bboxes_3d.tensor[:, :7].to(b.device).float()
            gl = g.labels_3d.to(b.device)
            if gb.shape[0] == 0:
                qls.append(F.binary_cross_entropy_with_logits(
                    o['q'], torch.zeros_like(o['q']), reduction='sum'))
                continue
            iou = box_iou_rotated(self._bev(b), self._bev(gb))   # [n, m]
            best, j = iou.max(1)
            same = (gl[None, :] == lab[:, None]).float()
            q_t = ((iou * same).max(1).values - 0.25).div(0.5).clamp(0, 1)
            qls.append(F.binary_cross_entropy_with_logits(
                o['q'], q_t, reduction='sum'))
            pos = best >= self.pos_iou
            if not pos.any():
                continue
            bp, gp = b[pos], gb[j[pos]]
            _, u, t = self._ray(bp)
            dc = gp[:, :2] - bp[:, :2]
            dyaw = (gp[:, 6] - bp[:, 6] + math.pi / 2) % math.pi - math.pi / 2
            tgt = torch.stack([
                (dc * u).sum(1) / self.range_scale, gp[:, 2] - bp[:, 2],
                (dc * t).sum(1) / self.cross_scale,
                *(gp[:, 3:6] / bp[:, 3:6]).clamp_min(1e-3).log().unbind(1),
                dyaw
            ], 1)
            pred = torch.stack([
                o['d_range'][pos], o['d_z'][pos], o['d_cross'][pos],
                *o['d_dim'][pos].unbind(1), o['d_yaw'][pos]
            ], 1)
            reg.append(F.smooth_l1_loss(pred, tgt, beta=0.1,
                                        reduction='none').sum(1))
            n_pos += int(pos.sum())
        n_all = sum(o['q'].numel() for o in outs if o is not None)
        dev = next(self.parameters()).device
        zero = sum(p.sum() for p in self.parameters()) * 0
        loss_reg = torch.cat(reg).sum() / max(n_pos, 1) if reg else zero
        loss_q = torch.stack(qls).sum() / max(n_all, 1) if qls else zero
        return dict(loss_refine_reg=self.w_reg * loss_reg + zero,
                    loss_refine_q=self.w_q * loss_q + zero,
                    refine_pos=torch.tensor(float(n_pos), device=dev))

    @staticmethod
    def _bev(b):
        return torch.stack([b[:, 0], b[:, 1], b[:, 3], b[:, 4], b[:, 6]], -1)


def image_branch(img_backbone, img_neck, imgs):
    """FPN features of every camera image, the per-camera brightness [B, N,
    2] (mean and std of the normalised band) and the image size."""
    img_hw = tuple(imgs.shape[-2:])
    x = imgs.flatten(0, 1) if imgs.dim() == 5 else imgs
    img_feats = img_neck(img_backbone(x))
    flat = imgs.flatten(2) if imgs.dim() == 5 else imgs[:, None].flatten(2)
    bright = torch.stack([flat.mean(-1), flat.std(-1)], -1)
    return img_feats, bright, img_hw


def refine_losses(rh, dets, gts, bev, img_feats, metas, bright, img_hw):
    """Losses of the second stage on the stage-1 detections ``dets``."""
    props, scores, labels = rh.sample_props(dets, gts)
    r = rh(props, scores, labels, bev, img_feats, metas, bright, img_hw)
    losses = rh.loss(r, props, labels, gts)
    if img_feats is not None:
        # keep the image branch in the graph when no proposal projects into
        # a camera (DDP needs a gradient for every parameter)
        losses['loss_refine_q'] = losses['loss_refine_q'] + \
            sum(f.sum() for f in img_feats) * 0
    return losses


def refine_predictions(rh, dets, bev, img_feats, metas, bright, img_hw):
    """Refined boxes and scores of the stage-1 detections ``dets``."""
    props = [d.bboxes_3d.tensor[:, :7] for d in dets]
    r = rh(props, [d.scores_3d for d in dets], [d.labels_3d for d in dets],
           bev, img_feats, metas, bright, img_hw)
    results = []
    for d, b, o, meta in zip(dets, props, r, metas):
        if o is None:
            results.append(d)
            continue
        out = InstanceData()
        out.bboxes_3d = meta['box_type_3d'](rh.refine_boxes(b.float(), o),
                                            box_dim=7)
        out.scores_3d = rh.score(d.scores_3d.float(), o)
        out.labels_3d = d.labels_3d
        results.append(out)
    return results


@MODELS.register_module()
class MariFusionCenterPoint(CenterPoint):
    """CenterPoint + RaySplitFusionRefineHead (see module docstring)."""

    def __init__(self, *args, refine_head: dict = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.refine_head = MODELS.build(refine_head)

    def _feats(self, inputs, metas):
        imgs = inputs.get('imgs', None)
        img_feats = bright = img_hw = None
        if imgs is not None and self.refine_head.use_image and \
                self.with_img_backbone:
            img_feats, bright, img_hw = image_branch(
                self.img_backbone, self.img_neck, imgs)
        pts_feats = self.extract_pts_feat(inputs.get('voxels'),
                                          points=inputs.get('points'),
                                          batch_input_metas=metas)
        return pts_feats, img_feats, bright, img_hw

    def _stage1(self, pts_feats, metas):
        outs = self.pts_bbox_head(pts_feats)
        det = [[{k: v.detach() for k, v in o[0].items()}] for o in outs]
        with torch.no_grad():
            dets = self.pts_bbox_head.predict_by_feat(det, metas)
        return outs, dets

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        gts = [s.gt_instances_3d for s in batch_data_samples]
        pts_feats, img_feats, bright, img_hw = self._feats(
            batch_inputs_dict, metas)
        outs, dets = self._stage1(pts_feats, metas)
        head = self.pts_bbox_head
        if hasattr(head, '_metas'):  # EvidenceSeaCenterHead: IMU tilt
            head._metas = metas
        try:
            losses = head.loss_by_feat(outs, gts)
        finally:
            if hasattr(head, '_metas'):
                head._metas = None
        losses.update(refine_losses(self.refine_head, dets, gts, pts_feats[0],
                                    img_feats, metas, bright, img_hw))
        return losses

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        metas = [s.metainfo for s in batch_data_samples]
        pts_feats, img_feats, bright, img_hw = self._feats(
            batch_inputs_dict, metas)
        _, dets = self._stage1(pts_feats, metas)
        results = refine_predictions(self.refine_head, dets, pts_feats[0],
                                     img_feats, metas, bright, img_hw)
        return self.add_pred_to_datasample(batch_data_samples, results)


@MODELS.register_module()
class BEVFuseCenterPoint(CenterPoint):
    """CenterPoint with BEVFusion's camera branch (baseline for MariFusion).

    The images go through ``img_backbone`` / ``img_neck``, the LSS
    ``view_transform`` lifts them to a BEV grid and ``fusion_layer``
    (ConvFuser) concatenates it with the sparse encoder's BEV before the
    2D backbone, exactly as projects/BEVFusion does; the LiDAR path and the
    CenterHead are unchanged, so the LiDAR checkpoint loads as is.
    """

    def __init__(self, *args, view_transform: dict = None,
                 fusion_layer: dict = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.view_transform = MODELS.build(view_transform)
        self.fusion_layer = MODELS.build(fusion_layer)

    def init_weights(self) -> None:
        # as projects/BEVFusion: the LSS transform holds a Long parameter
        # that mmengine's generic init_weights cannot average
        self.img_backbone.init_weights()

    def extract_img_feat(self, img, input_metas):
        if img is None:
            return None
        B = img.shape[0]
        x = img.flatten(0, 1) if img.dim() == 5 else img
        x = self.img_neck(self.img_backbone(x))
        if not isinstance(x, torch.Tensor):
            x = x[0]
        return x.view(B, -1, *x.shape[1:])

    def extract_feat(self, batch_inputs_dict, batch_input_metas):
        # the image features are consumed inside the LiDAR path (fused
        # before the 2D backbone); MVXTwoStageDetector's loss / predict
        # would otherwise look for an image detection head
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'),
                                          batch_input_metas)
        pts_feats = self.extract_pts_feat(
            batch_inputs_dict['voxels'], points=batch_inputs_dict['points'],
            img_feats=img_feats, batch_input_metas=batch_input_metas)
        return None, pts_feats

    def extract_pts_feat(self, voxel_dict, points=None, img_feats=None,
                         batch_input_metas=None):
        vf = self.pts_voxel_encoder(voxel_dict['voxels'],
                                    voxel_dict['num_points'],
                                    voxel_dict['coors'])
        bs = voxel_dict['coors'][-1, 0] + 1
        x = self.pts_middle_encoder(vf, voxel_dict['coors'], bs)
        if img_feats is not None:
            def mat(key, default=None):
                rows = [m[key] if default is None else m.get(key, default)
                        for m in batch_input_metas]
                return x.new_tensor(np.asarray(rows, dtype=np.float32))
            with torch.autocast('cuda', enabled=False):
                img_bev = self.view_transform(
                    img_feats.float(), [p.float() for p in points],
                    mat('lidar2img'), mat('cam2img'), mat('cam2lidar'),
                    mat('img_aug_matrix', np.eye(4)),
                    mat('lidar_aug_matrix', np.eye(4)), batch_input_metas)
            x = self.fusion_layer([img_bev, x])
        x = self.pts_backbone(x)
        return self.pts_neck(x)
