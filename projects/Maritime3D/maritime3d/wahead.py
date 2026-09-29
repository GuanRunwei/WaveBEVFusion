"""WAHead: waterline-anchored detection head for floating targets.

Two additions on top of TransFusionHead (projects/BEVFusion):

VWA -- visible-waterline anchor. The heatmap peak and the query's reference
point sit on the sensor-facing side of the object,

    a = c - kappa * e(l, w, yaw, theta) * r_hat(theta),
    e = l/2 * |cos(yaw - theta)| + w/2 * |sin(yaw - theta)|,

and not at the geometric centre c, which for large vessels lies on empty
water behind the hull (the LiDAR centroid sits 1.9 m in front of the centre
for boats and 7.8 m for ships). The network regresses the anchor, and the
centre is decoded geometrically from (anchor, size, yaw). The shift runs along
the viewing ray, so the anchor's azimuth equals the centre's and the decode is
exact. kappa = 0 recovers the plain centre heatmap.

SSR -- shared sea-surface reasoning. Every floating object rests on the same
sea surface. Each query proposes its own waterline point (decoded centre xy,
raw bottom z). A score-weighted least-squares fit with a learned global prior
plane gives this frame's plane z = a*x + b*y + c, and
each box's final bottom is plane(x, y) + a small learned freeboard residual.
The plane is supervised with a robust fit to the (augmented) GT bottoms, so
no extra labels are needed. Oracle on this data: sharing the plane halves the
median |dz| (0.82 m -> 0.37 m).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models import draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS, TASK_UTILS
from projects.BEVFusion.bevfusion.transfusion_head import TransFusionHead
from projects.BEVFusion.bevfusion.utils import TransFusionBBoxCoder

LOG_DIM_RANGE = (-5.0, 6.0)  # exp -> 7 mm .. 403 m; keeps early training finite


def ray_half_extent(l, w, yaw, theta):
    """Half-extent of a (l, w, yaw) box along the viewing direction theta."""
    d = yaw - theta
    return 0.5 * l * torch.cos(d).abs() + 0.5 * w * torch.sin(d).abs()


def center_to_anchor(x, y, l, w, yaw, kappa, max_frac=0.9):
    """Move the centre towards the sensor by kappa * half-extent."""
    r = torch.sqrt(x * x + y * y).clamp(min=1e-3)
    theta = torch.atan2(y, x)
    shift = torch.minimum(kappa * ray_half_extent(l, w, yaw, theta),
                          max_frac * r)
    return x - shift * x / r, y - shift * y / r


def anchor_to_center(ax, ay, l, w, yaw, kappa):
    """Inverse of center_to_anchor (exact unless the clamp was active)."""
    r = torch.sqrt(ax * ax + ay * ay).clamp(min=1e-3)
    theta = torch.atan2(ay, ax)
    shift = kappa * ray_half_extent(l, w, yaw, theta)
    return ax + shift * ax / r, ay + shift * ay / r


@TASK_UTILS.register_module()
class WaterlineBBoxCoder(TransFusionBBoxCoder):
    """TransFusionBBoxCoder whose BEV 'center' code is the VWA anchor."""

    def __init__(self, *args, kappa=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = float(kappa)

    def _cell(self, i):
        return self.out_size_factor * self.voxel_size[i]

    def encode(self, dst_boxes):
        if self.kappa == 0:
            return super().encode(dst_boxes)
        b = dst_boxes.clone()
        ax, ay = center_to_anchor(b[:, 0], b[:, 1], b[:, 3], b[:, 4],
                                  b[:, 6], self.kappa)
        b[:, 0], b[:, 1] = ax, ay
        return super().encode(b)

    def anchor_to_center_cells(self, center, dim, rot):
        """[B, 2, P] anchor (cells) -> [B, 2, P] centre (cells).

        dim is the log-size code [B, 3, P] and rot is (sin, cos) [B, 2, P].
        Differentiable, so a loss on the decoded centre reaches size and
        yaw too.
        """
        if self.kappa == 0:
            return center
        s0, s1 = self._cell(0), self._cell(1)
        ax = center[:, 0] * s0 + self.pc_range[0]
        ay = center[:, 1] * s1 + self.pc_range[1]
        logd = dim.clamp(*LOG_DIM_RANGE)
        l, w = logd[:, 0].exp(), logd[:, 1].exp()
        yaw = torch.atan2(rot[:, 0], rot[:, 1])
        cx, cy = anchor_to_center(ax, ay, l, w, yaw, self.kappa)
        return torch.stack(
            [(cx - self.pc_range[0]) / s0, (cy - self.pc_range[1]) / s1],
            dim=1)

    def decode(self, heatmap, rot, dim, center, height, vel, filter=False):
        center = self.anchor_to_center_cells(center, dim, rot)
        return super().decode(heatmap, rot, dim, center, height, vel, filter)


def fit_plane_torch(pts, scale, min_boxes=3, min_spread=3.0, clip=1.5,
                    n_iter=3):
    """Robust plane z = a*(x/s) + b*(y/s) + c through [N, 3] points.

    Returns a [3] tensor, or None when under-determined (fewer than
    min_boxes points, or nearly collinear in xy). Mirrors fit_sea_plane in
    tools/create_maritime_infos_from_tables.py.
    """
    n = pts.shape[0]
    if n < min_boxes:
        return None
    xy = pts[:, :2] - pts[:, :2].mean(0, keepdim=True)
    if torch.linalg.svdvals(xy)[-1] / n**0.5 < min_spread:
        return None
    X = torch.stack([pts[:, 0] / scale, pts[:, 1] / scale,
                     torch.ones_like(pts[:, 0])], 1)
    keep = torch.ones(n, dtype=torch.bool, device=pts.device)
    coef = None
    eye = torch.eye(3, device=pts.device) * 1e-6
    for _ in range(n_iter):
        if int(keep.sum()) < min_boxes:
            break
        Xk, zk = X[keep], pts[keep, 2]
        coef = torch.linalg.solve(Xk.T @ Xk + eye, Xk.T @ zk)
        new_keep = (X @ coef - pts[:, 2]).abs() < clip
        if bool((new_keep == keep).all()):
            break
        keep = new_keep
    if coef is None or int(keep.sum()) < min_boxes:
        return None
    return coef


@MODELS.register_module()
class WAHead(TransFusionHead):
    """TransFusionHead + VWA anchors + SSR shared sea surface.

    Use with ``bbox_coder=dict(type='WaterlineBBoxCoder', kappa=...)``
    (kappa=0 turns VWA off). SSR needs a freeboard entry
    ``common_heads['fb'] = [1, 2]``.

    Args:
        ssr (bool): Enable shared sea-surface reasoning.
        plane_scale (float): Metres per unit of the plane's x/y terms, which
            keeps the 3x3 solve well conditioned over a +-160 m range.
        plane_prior_lambda (tuple): Ridge strength pulling (a, b, c) towards
            the learned global prior plane, in units of summed query weight.
            Frames with fewer than ~3 confident objects fall back to the
            prior.
        plane_prior_c (float): Initial prior waterline height in the LiDAR
            frame. The median GT box bottom is about -1.3 m.
        loss_plane_weight (float): L1 on (a, b, c) against a robust fit to the
            GT bottoms, on frames where that fit is well posed.
        loss_height_raw_weight (float): Auxiliary L1 on each query's own
            waterline estimate (the pre-SSR height), so every query keeps
            learning its own height.
        loss_center_weight (float): L1 on the geometrically decoded centre,
            which couples the anchor, size and yaw errors (VWA only).
        temporal (bool): Temporal SSR. At test time the plane is a
            recursive weighted least-squares fit: each frame's evidence
            (X^T W X, X^T W z) is added to the sequence's history, decayed
            by ``temporal_forget`` per 0.1 s, so many frames of weak
            evidence add up (consecutive frames move the plane by a median
            3 cm / 0.06 deg). The stream is keyed on the sequence and
            timestamp parsed from ``lidar_path`` (points4/<seq>/<ns>.bin) and
            resets after ``temporal_max_gap`` seconds; evaluation must visit
            each sequence in time order (SequentialChunkSampler). In
            training, which sees shuffled single frames, the history is
            simulated with probability ``temporal_prob`` as a pseudo
            observation of the GT plane plus noise ``temporal_noise`` (a, b in
            m per ``plane_scale``, c in m) with strength U(0,
            ``temporal_max_weight``).
        ssr_decouple (bool): Stop the gradient of the final height at the
            per-query raw heights and sizes, both in the plane fit and in the
            ``+ h / 2`` term, so one box's height error cannot reshape other
            boxes (or its own size) through the shared plane; the final-height
            loss then trains the freeboard and the prior plane only.
        imu_tilt (bool): Take the plane's tilt (a, b) from the vessel's IMU
            instead of fitting it: the LiDAR is body-fixed, so the sea tilts
            in its frame with the hull's attitude (up to ~4 deg under way),
            while the tilt of a plane fitted to annotated box bottoms is
            mostly label noise (~0.7 deg). Needs the ``sea_up`` /
            ``sea_up_valid`` metas from :class:`LoadSeaUp` (gravity up in the
            augmented LiDAR frame). A learned constant ``tilt_offset`` absorbs
            the mounting bias; frames without IMU fall back to the prior's
            tilt. Only the sea level c is then solved from the queries (and
            carried by the temporal stream), and ``loss_plane`` supervises c
            alone against the GT bottoms measured under the same tilt, which
            needs one box instead of three.
        tilt_gain (bool): With ``imu_tilt``, scale the IMU tilt per body axis
            (pitch, roll) by a learned gain g = 1 + delta before the mounting
            offset, in the vessel's frame (undoing the augmentation with the
            ``sea_up_rot`` meta). The annotated box heights follow the hull's
            attitude only partly (work_dirs/tmp/horizon/heave/report.txt), so
            the physically right tilt does not match the labels the detector
            is scored on; the learned g measures by how much. Weight decay
            pulls g towards 1, the physical tilt.
        sea_field (bool): Replace the rigid shared plane by a physical sea
            model (needs ``imu_tilt`` and ``common_heads['sig']``). The sea
            height under query i is
                s_i = tilt(x_i, y_i) + c + eta(x_i, y_i),
            with the tilt from the IMU, the mean level c ~ N(c-, P-) from a
            one-step Kalman prior, and eta a zero-mean Gaussian-process wave
            field, k = sw^2 * l^2 / lij^2 * exp(-d^2 / 2 lij^2),
            lij^2 = l^2 + (L_i^2 + L_j^2) / 12, i.e. waves averaged over each
            hull's length L (long ships ride over them). The top
            ``field_obs`` queries observe their bottom y_j = s_j + e_j,
            e_j ~ N(0, 2 b_j^2 / score_j), where b_j is a learned Laplace
            scale (``sig`` head, Laplace NLL on the matched queries). One
            closed-form solve gives the Kalman update of c (with
            wave-correlated observation noise) and the kriged sea height under
            every query: a well-observed hull keeps its own waterline, a
            sparse far one borrows from neighbours within ~l, an isolated one
            falls back to the mean sea; sw -> 0 recovers the rigid plane. The
            wave amplitude and correlation length and the no-history level
            variance are learned (log-parametrised). The Kalman prior is
            simulated in training (GT level + noise of random std in
            ``hist_std_range``, with probability ``temporal_prob``) and
            streamed at test time: c- = c_prev - ``heave_gain`` * dheave,
            P- = P_prev + ``kalman_q`` per 0.1 s. Observations are detached,
            as with ``ssr_decouple``.
        sea_context (bool): Feed the plane back into the next decoder layer:
            each query gets an embedding of (its own bottom minus the sea
            height under it, a, b, c) added to its features. Needs
            ``num_decoder_layers`` >= 2; the embedding starts at zero.
    """

    def __init__(self,
                 *args,
                 ssr=True,
                 plane_scale=100.0,
                 plane_prior_lambda=(1.0, 1.0, 1.0),
                 plane_prior_c=-1.3,
                 loss_plane_weight=0.5,
                 loss_height_raw_weight=0.25,
                 loss_center_weight=0.25,
                 temporal=False,
                 temporal_forget=0.9,
                 temporal_max_gap=1.0,
                 temporal_prob=0.5,
                 temporal_noise=(0.5, 0.5, 0.3),
                 temporal_max_weight=10.0,
                 sea_context=False,
                 ssr_decouple=False,
                 imu_tilt=False,
                 tilt_gain=False,
                 sea_field=False,
                 field_obs=50,
                 wave_sigma=0.3,
                 wave_length=25.0,
                 level_prior_var=1.0,
                 hist_std_range=(0.05, 1.0),
                 kalman_q=0.0025,
                 heave_gain=1.0,
                 loss_sigma_weight=0.25,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.ssr = ssr
        self.plane_scale = float(plane_scale)
        self.loss_plane_weight = loss_plane_weight
        self.loss_height_raw_weight = loss_height_raw_weight
        self.loss_center_weight = loss_center_weight
        self.kappa = float(getattr(self.bbox_coder, 'kappa', 0.0))
        self._cached_targets = None
        self.temporal = temporal and ssr
        self.temporal_forget = float(temporal_forget)
        self.temporal_max_gap = float(temporal_max_gap)
        self.temporal_prob = float(temporal_prob)
        self.temporal_noise = tuple(temporal_noise)
        self.temporal_max_weight = float(temporal_max_weight)
        self._stream = {}  # test time: seq -> (timestamp_ns, A, b)
        self._train_planes = None  # training: per-sample GT plane or None
        self.ssr_decouple = ssr_decouple
        self.imu_tilt = imu_tilt and ssr
        if self.imu_tilt:
            self.tilt_offset = nn.Parameter(torch.zeros(2))
        self.tilt_gain = tilt_gain and self.imu_tilt
        if self.tilt_gain:
            # gain = 1 + delta, so weight decay pulls towards the IMU tilt
            self.tilt_gain_delta = nn.Parameter(torch.zeros(2))
        self.sea_field = sea_field and ssr
        if self.sea_field:
            assert self.imu_tilt, 'sea_field takes the mean tilt from the IMU'
            assert 'sig' in self.prediction_heads[0].heads, \
                "sea_field needs common_heads['sig'] = [1, 2]"
            self.field_obs = int(field_obs)
            self.log_wave_sigma = nn.Parameter(
                torch.tensor(float(wave_sigma)).log())
            self.log_wave_length = nn.Parameter(
                torch.tensor(float(wave_length)).log())
            self.log_level_prior_var = nn.Parameter(
                torch.tensor(float(level_prior_var)).log())
            self.hist_std_range = tuple(float(v) for v in hist_std_range)
            self.kalman_q = float(kalman_q)
            self.heave_gain = float(heave_gain)
            self.loss_sigma_weight = float(loss_sigma_weight)
            self._stream_field = {}  # seq -> (ts, c, P, heave or None)
        self.sea_context = sea_context and ssr
        if self.sea_context:
            assert self.num_decoder_layers >= 2, \
                'sea_context feeds the plane to a later decoder layer'
            hidden = kwargs.get('hidden_channel', 128)
            self.sea_embed = nn.Sequential(
                nn.Conv1d(4, hidden, 1), nn.ReLU(inplace=True),
                nn.Conv1d(hidden, hidden, 1))
            nn.init.zeros_(self.sea_embed[-1].weight)
            nn.init.zeros_(self.sea_embed[-1].bias)
        if ssr:
            assert 'fb' in self.prediction_heads[0].heads, \
                "SSR needs common_heads['fb'] = [1, 2]"
            # A learned global plane, not an MLP on pooled BEV features: the
            # backbone's BN statistics differ between train and eval on this
            # large, mostly empty BEV, and the pooled-feature prior drifted
            # from c = -2.5 m (train mode) to -10.7 m (eval mode), dragging
            # every box ~3.8 m down (work_dirs/bench_tfl_wahead).
            self.plane_prior = nn.Parameter(
                torch.tensor([0.0, 0.0, float(plane_prior_c)]))
            self.register_buffer(
                'plane_lambda',
                torch.tensor(plane_prior_lambda, dtype=torch.float32),
                persistent=False)

    # ------------------------------------------------------------ helpers
    def _cells_to_m(self, cxy):
        c = self.bbox_coder
        return (cxy[:, 0] * c._cell(0) + c.pc_range[0],
                cxy[:, 1] * c._cell(1) + c.pc_range[1])

    def _plane_z(self, plane, x, y):
        s = self.plane_scale
        return plane[:, 0:1] * x / s + plane[:, 1:2] * y / s + plane[:, 2:3]

    def _evidence(self, cxy, dim, h_raw, weight, tilt=None):
        """Per-frame normal equations of z = a*x/s + b*y/s + c.

        cxy: decoded centres in cells [B, 2, P], detached. dim: log sizes
        [B, 3, P]. h_raw: raw gravity-centre z [B, 1, P]. weight: [B, P].
        Returns (X^T W X [B, k, k], X^T W z [B, k]) over the queries'
        waterline points: k = 3 for (a, b, c), or k = 1 for c alone when the
        tilt [B, 2] is given (the IMU case).
        """
        x, y = self._cells_to_m(cxy)
        h = dim[:, 2].clamp(*LOG_DIM_RANGE).exp()
        bottom = h_raw[:, 0] - 0.5 * h
        s = self.plane_scale
        if tilt is None:
            X = torch.stack([x / s, y / s, torch.ones_like(x)], dim=-1)
        else:
            X = torch.ones_like(x)[..., None]
            bottom = bottom - (tilt[:, 0:1] * x + tilt[:, 1:2] * y) / s
        XtW = X.transpose(1, 2) * weight[:, None, :]
        return XtW @ X, (XtW @ bottom[..., None])[..., 0]

    def _solve_plane(self, A, b, prior, hist_A=None, hist_b=None):
        """Ridge solution (A + H + L) p = b + h + L prior.

        A, b: this frame's evidence; H, h: history (temporal SSR) or None;
        L: the global prior strength ``plane_prior_lambda``. prior [B, k]
        holds the solved parameters only, i.e. (a, b, c) or (c, ).
        """
        lam = self.plane_lambda[-prior.shape[-1]:].to(prior)
        A = A + torch.diag_embed(lam.expand_as(prior))
        b = b + lam * prior
        if hist_A is not None:
            A, b = A + hist_A, b + hist_b
        return torch.linalg.solve(A, b)

    # ------------------------------------------------------ temporal SSR
    @staticmethod
    def _stream_key(meta):
        """(seq, timestamp in ns) from points4/<seq>/<ns>.bin, or None."""
        path = str(meta.get('lidar_path', ''))
        parts = path.replace('\\', '/').split('/')
        try:
            return parts[-2], int(parts[-1].split('.')[0])
        except (IndexError, ValueError):
            return None

    def _history(self, metas, like):
        """History terms (H [B, k, k], h [B, k]) for this batch, or None.

        ``like`` is the prior of the solved parameters, [B, k].
        """
        B, k = like.shape
        H = like.new_zeros(B, k, k)
        h = like.new_zeros(B, k)
        if self.training:
            if self._train_planes is None:
                return None
            noise = like.new_tensor(self.temporal_noise[-k:])
            for i, gt in enumerate(self._train_planes):
                if gt is None or torch.rand(()) >= self.temporal_prob:
                    continue
                w = float(torch.rand(())) * self.temporal_max_weight
                target = gt.to(like) + noise * torch.randn(k, device=like.device)
                H[i] = w * torch.eye(k, device=like.device)
                h[i] = w * target
            return H, h
        for i, meta in enumerate(metas):
            key = self._stream_key(meta)
            if key is None or key[0] not in self._stream:
                continue
            ts, sA, sb = self._stream[key[0]]
            dt = (key[1] - ts) * 1e-9
            if 0 < dt <= self.temporal_max_gap:
                decay = self.temporal_forget**(dt / 0.1)
                H[i], h[i] = decay * sA.to(like), decay * sb.to(like)
        return H, h

    def _update_stream(self, metas, A, b, H, h):
        """Store this frame's accumulated evidence (test time)."""
        for i, meta in enumerate(metas):
            key = self._stream_key(meta)
            if key is not None:
                self._stream[key[0]] = (key[1], (A[i] + H[i]).detach(),
                                        (b[i] + h[i]).detach())

    def _imu_tilt(self, metas, prior):
        """Plane tilt (a, b) [B, 2] from the gravity-up vector in the metas.

        a = -s * up_x / up_z, b = -s * up_y / up_z, times the learned
        per-body-axis gain (``tilt_gain``), plus the learned mounting offset;
        frames without a valid IMU reading take the prior's tilt.
        """
        s = self.plane_scale
        # keep every tilt parameter in the graph: a rank whose whole batch
        # lacks IMU would otherwise leave them without a gradient and stop
        # DDP (bench_gn_ssr_imu crashed on this at ep5)
        keep = 0 * self.tilt_offset.float()
        gain = None
        if self.tilt_gain:
            gain = 1 + self.tilt_gain_delta.float()
            keep = keep + 0 * gain
        rows = []
        for i, meta in enumerate(metas):
            up = meta.get('sea_up')
            if up is not None and bool(meta.get('sea_up_valid', False)):
                up = prior.new_tensor(up)
                ab = torch.stack([-s * up[0] / up[2], -s * up[1] / up[2]])
                if gain is not None:
                    rot = meta.get('sea_up_rot')
                    if rot is None:
                        assert not self.training, \
                            'tilt_gain needs the sea_up_rot meta in training'
                        rot = prior.new_tensor([[1.0, 0.0], [0.0, 1.0]])
                    rot = prior.new_tensor(rot)
                    # body frame -> scale pitch / roll -> augmented frame
                    ab = rot @ (gain * torch.linalg.solve(rot, ab))
                rows.append(ab + self.tilt_offset.float() + keep)
            else:
                rows.append(prior[i, :2] + keep)
        return torch.stack(rows)

    def _gt_sea(self, boxes, tilt=None):
        """GT sea plane of one frame, in the solved parameters, or None.

        Without a tilt: a robust (a, b, c) fit to the box bottoms (>= 3 boxes,
        not collinear). With the IMU tilt: c alone, the median bottom after
        removing the tilt (>= 1 box).
        """
        pts = boxes[:, :3].float()
        if tilt is None:
            return fit_plane_torch(pts, self.plane_scale) \
                if pts.shape[0] >= 3 else None
        if pts.shape[0] == 0:
            return None
        t = tilt.detach().to(pts)
        level = pts[:, 2] - (t[0] * pts[:, 0] + t[1] * pts[:, 1]) \
            / self.plane_scale
        return level.median()[None]

    def loss(self, batch_feats, batch_data_samples):
        """GT sea planes first: temporal SSR simulates its history from
        them, and with ``imu_tilt`` the plane loss supervises their c."""
        if self.temporal or self.imu_tilt:
            dev = self.plane_prior.device
            tilts = None
            if self.imu_tilt:
                metas = [s.metainfo for s in batch_data_samples]
                tilts = self._imu_tilt(
                    metas, self.plane_prior.float().expand(len(metas), 3))
            self._train_planes = [
                self._gt_sea(s.gt_instances_3d.bboxes_3d.tensor.to(dev),
                             None if tilts is None else tilts[i])
                for i, s in enumerate(batch_data_samples)
            ]
        try:
            return super().loss(batch_feats, batch_data_samples)
        finally:
            self._train_planes = None

    # ------------------------------------------------- sea field (Kalman+GP)
    LOG_SIG_RANGE = (-3.0, 2.0)  # Laplace scale of a bottom: 5 cm .. 7.4 m

    def _level_prior(self, metas, like):
        """One-step Kalman prior (c-, P-) of the mean sea level, [B] each.

        Training (option A): with probability ``temporal_prob`` the history
        is simulated as the GT level plus noise of a random std s, with
        P- = s^2; otherwise the learned no-history prior (c0, P0). Test: the
        streamed posterior of the previous frame of the same sequence,
        propagated by the IMU heave and the process noise ``kalman_q``.
        """
        B = len(metas)
        c0 = self.plane_prior[2].float()
        P0 = self.log_level_prior_var.float().exp()
        c, P = [], []
        lo, hi = self.hist_std_range
        for i, meta in enumerate(metas):
            ci, Pi = c0, P0
            if self.training:
                gt = None if self._train_planes is None \
                    else self._train_planes[i]
                if gt is not None and torch.rand(()) < self.temporal_prob:
                    s = lo + (hi - lo) * float(torch.rand(()))
                    ci = gt.to(like).reshape(()) + \
                        s * torch.randn((), device=like.device)
                    Pi = like.new_tensor(s * s)
            else:
                key = self._stream_key(meta)
                prev = self._stream_field.get(key[0]) if key else None
                if prev is not None:
                    ts, cp, Pp, hp = prev
                    dt = (key[1] - ts) * 1e-9
                    if 0 < dt <= self.temporal_max_gap:
                        h = meta.get('sea_heave')
                        dh = 0.0
                        if hp is not None and h is not None and \
                                bool(meta.get('sea_heave_valid', False)):
                            dh = float(h) - hp
                        ci = like.new_tensor(cp - self.heave_gain * dh)
                        Pi = like.new_tensor(Pp + self.kalman_q * dt / 0.1)
            c.append(ci)
            P.append(Pi)
        # keep c0 / P0 in the graph when every sample had a (simulated)
        # history, or DDP stops on parameters without a gradient
        keep = 0 * (c0 + P0)
        return torch.stack(c) + keep, torch.stack(P) + keep

    def _update_field_stream(self, metas, c, P):
        """Store this frame's posterior level (test time)."""
        for i, meta in enumerate(metas):
            key = self._stream_key(meta)
            if key is None:
                continue
            h = meta.get('sea_heave')
            ok = h is not None and bool(meta.get('sea_heave_valid', False))
            self._stream_field[key[0]] = (key[1], float(c[i]), float(P[i]),
                                          float(h) if ok else None)

    def _wave_cov(self, xa, ya, La, xb, yb, Lb):
        """Wave covariance between hull-footprint-averaged sea heights.

        x*, y*: [B, n] metres; L*: [B, n] hull lengths. Isotropic Gaussian
        smoothing of variance L^2/12 per hull keeps the kernel positive
        definite: k = sw^2 * l^2 / lij^2 * exp(-d^2 / (2 lij^2)).
        """
        sw2 = (2 * self.log_wave_sigma.float()).exp()
        l2 = (2 * self.log_wave_length.float()).exp()
        lij2 = l2 + (La[:, :, None]**2 + Lb[:, None, :]**2) / 12.0
        d2 = (xa[:, :, None] - xb[:, None, :])**2 + \
            (ya[:, :, None] - yb[:, None, :])**2
        return sw2 * (l2 / lij2) * torch.exp(-0.5 * d2 / lij2)

    def _ssr_field(self, res, tilt, c_prior, P_prior):
        """Kalman update of the mean level + GP kriging of the wave field.

        tilt: [B, 2] IMU tilt; c_prior, P_prior: [B]. Returns the final
        gravity-centre height [B, 1, P], the plane (a, b, c+) [B, 3], the
        posterior (c+, P+) [B] each and the sea context [B, 4, P].
        """
        center = res['center'].float()
        dim = res['dim'].float().detach()
        rot = res['rot'].float()
        h_raw = res['height'].float().detach()
        fb = res['fb'].float()
        log_b = res['sig'].float()[:, 0].clamp(*self.LOG_SIG_RANGE)
        score = res['heatmap'].detach().float().sigmoid().max(1).values
        cxy = self.bbox_coder.anchor_to_center_cells(center, dim, rot).detach()
        x, y = self._cells_to_m(cxy)                           # [B, P]
        logd = dim.clamp(*LOG_DIM_RANGE)
        L, h = logd[:, 0].exp(), logd[:, 2].exp()
        tilt_z = (tilt[:, 0:1] * x + tilt[:, 1:2] * y) / self.plane_scale
        bottom = h_raw[:, 0] - 0.5 * h
        # observations: the top-M queries by score, noise 2 b^2 / score
        M = min(self.field_obs, x.shape[1])
        idx = score.topk(M, dim=1).indices

        def g(t):
            return t.gather(1, idx)

        xo, yo, Lo = g(x), g(y), g(L)
        innov = g(bottom - tilt_z) - c_prior[:, None]
        R = 2 * (2 * g(log_b)).exp() / g(score).clamp_min(1e-3)
        Pm = P_prior[:, None, None]
        S = Pm + self._wave_cov(xo, yo, Lo, xo, yo, Lo) + torch.diag_embed(R)
        S = S + 1e-4 * torch.eye(M, device=S.device)
        sol = torch.linalg.solve(
            S, torch.stack([innov, torch.ones_like(innov)], dim=-1))
        alpha, s_inv1 = sol[..., 0], sol[..., 1]               # [B, M]
        c_post = c_prior + P_prior * alpha.sum(1)
        P_post = P_prior - P_prior**2 * s_inv1.sum(1)
        # kriging: cov(sea_i, y_O) = P- + k_wave(i, O)
        Kx = Pm + self._wave_cov(x, y, L, xo, yo, Lo)          # [B, P, M]
        sea = tilt_z + c_prior[:, None] + (Kx @ alpha[..., None])[..., 0]
        # final gravity-centre z = sea(x, y) + freeboard + h / 2
        height = (sea + 0.5 * h)[:, None] + fb
        plane = torch.cat([tilt, c_post[:, None]], dim=1)
        ctx = torch.cat([
            (bottom - sea)[:, None].clamp(-5, 5),
            plane[:, :, None].expand(-1, -1, sea.shape[-1])
        ], dim=1).detach()
        return height, plane, (c_post, P_post), ctx

    # ------------------------------------------------------------ forward
    def _ssr(self, res, prior, hist, tilt=None):
        """Shared sea surface for one decoder layer's predictions.

        prior: [B, 3] global prior plane; hist: history of the solved
        parameters or None; tilt: [B, 2] IMU tilt or None (then all of
        (a, b, c) are solved). Returns the final gravity-centre height
        [B, 1, P], the plane [B, 3], this frame's evidence (A, b) and the
        per-query sea context [B, 4, P].
        """
        center = res['center'].float()
        dim = res['dim'].float()
        rot = res['rot'].float()
        h_raw = res['height'].float()
        fb = res['fb'].float()
        weight = res['heatmap'].detach().float().sigmoid().max(1).values**2
        cxy = self.bbox_coder.anchor_to_center_cells(center, dim, rot).detach()
        if self.ssr_decouple:
            # the final-height loss then reaches only fb and the prior;
            # h and the raw heights keep their own losses, as in the
            # baseline (bench_tfl_ssr ep5 without this: h ratio 0.82,
            # bottoms +0.75 m)
            dim, h_raw = dim.detach(), h_raw.detach()
        A, b = self._evidence(cxy, dim, h_raw, weight, tilt)
        if tilt is None:
            plane = self._solve_plane(A, b, prior, *(hist or (None, None)))
        else:
            c = self._solve_plane(A, b, prior[:, 2:], *(hist or (None, None)))
            plane = torch.cat([tilt, c], dim=1)
        x, y = self._cells_to_m(cxy)
        h = dim[:, 2].clamp(*LOG_DIM_RANGE).exp()
        sea = self._plane_z(plane, x, y)
        # final gravity-centre z = plane(x, y) + freeboard + h / 2
        height = (sea + 0.5 * h)[:, None] + fb
        ctx = torch.cat([
            (h_raw[:, 0] - 0.5 * h - sea)[:, None].clamp(-5, 5),
            plane[:, :, None].expand(-1, -1, sea.shape[-1])
        ], dim=1).detach()
        return height, plane, (A, b), ctx

    def forward_single(self, inputs, metas):
        """TransFusionHead.forward_single with SSR after every layer.

        The query initialisation is the parent's, verbatim; the decoder loop
        solves the sea plane from each layer's predictions and, with
        ``sea_context``, feeds it into the next layer.
        """
        if not self.ssr:
            return super().forward_single(inputs, metas)
        batch_size = inputs.shape[0]
        fusion_feat = self.shared_conv(inputs)
        fusion_feat_flatten = fusion_feat.view(batch_size,
                                               fusion_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(fusion_feat.device)

        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head(fusion_feat.float())
        heatmap = dense_heatmap.detach().sigmoid()
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                  padding:(-padding)] = local_max_inner
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = fusion_feat_flatten.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, fusion_feat_flatten.shape[1], -1),
            dim=-1)
        self.query_labels = top_proposals_class
        one_hot = F.one_hot(
            top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_feat += self.class_encoding(one_hot.float())
        query_pos = bev_pos.gather(
            index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(
                -1, -1, bev_pos.shape[-1]),
            dim=1)

        with torch.autocast('cuda', enabled=False):
            prior = self.plane_prior.float().expand(batch_size, 3)
            tilt = self._imu_tilt(metas, prior) if self.imu_tilt else None
            solved = prior if tilt is None else prior[:, 2:]
            if self.sea_field:
                level = self._level_prior(metas, prior)
                hist = None
            else:
                hist = self._history(metas, solved) if self.temporal \
                    else None
        ret_dicts, ctx = [], None
        for i in range(self.num_decoder_layers):
            if ctx is not None and self.sea_context:
                query_feat = query_feat + self.sea_embed(ctx).to(
                    query_feat.dtype)
            query_feat = self.decoder[i](
                query_feat,
                key=fusion_feat_flatten,
                query_pos=query_pos,
                key_pos=bev_pos)
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(
                0, 2, 1)
            with torch.autocast('cuda', enabled=False):
                if self.sea_field:
                    height, plane, evidence, ctx = self._ssr_field(
                        res_layer, tilt, *level)
                else:
                    height, plane, evidence, ctx = self._ssr(
                        res_layer, prior, hist, tilt)
            res_layer['height_raw'] = res_layer['height']
            res_layer['height'] = height
            res_layer['plane'] = plane[..., None]  # [B, 3, 1]
            ret_dicts.append(res_layer)
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)
        if self.sea_field and not self.training:
            self._update_field_stream(metas, *evidence)
        elif self.temporal and not self.training:
            self._update_stream(metas, *evidence, *hist)

        ret_dicts[0]['query_heatmap_score'] = heatmap.gather(
            index=top_proposals_index[:, None, :].expand(
                -1, self.num_classes, -1),
            dim=-1)
        ret_dicts[0]['dense_heatmap'] = dense_heatmap
        if self.auxiliary is False:
            return [ret_dicts[-1]]
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ('dense_heatmap', 'dense_heatmap_old',
                           'query_heatmap_score'):
                new_res[key] = torch.cat([r[key] for r in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    # ------------------------------------------------------------ targets
    def get_targets(self, batch_gt_instances_3d, preds_dict):
        res = super().get_targets(batch_gt_instances_3d, preds_dict)
        # loss_by_feat of the parent does not hand the targets back; keep
        # them for the extra losses computed right after it
        self._cached_targets = res
        return res

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        res = super().get_targets_single(gt_instances_3d, preds_dict,
                                         batch_idx)
        if self.kappa == 0:
            return res
        return res[:-1] + (self._anchor_heatmap(gt_instances_3d, res[-1]), )

    def _anchor_heatmap(self, gt_instances_3d, like):
        """Same Gaussians as the parent, but peaked at the VWA anchor."""
        heatmap = torch.zeros_like(like[0])
        boxes = gt_instances_3d.bboxes_3d.tensor.to(like.device)
        labels = gt_instances_3d.labels_3d
        if boxes.shape[0] == 0:
            return heatmap[None]
        cfg = self.train_cfg
        vs, osf = cfg['voxel_size'], cfg['out_size_factor']
        pcr = cfg['point_cloud_range']
        ax, ay = center_to_anchor(boxes[:, 0], boxes[:, 1], boxes[:, 3],
                                  boxes[:, 4], boxes[:, 6], self.kappa)
        for i in range(boxes.shape[0]):
            width = boxes[i, 3] / vs[0] / osf
            length = boxes[i, 4] / vs[1] / osf
            if width > 0 and length > 0:
                radius = gaussian_radius((length, width),
                                         min_overlap=cfg['gaussian_overlap'])
                radius = max(cfg['min_radius'], int(radius))
                cx = (ax[i] - pcr[0]) / vs[0] / osf
                cy = (ay[i] - pcr[1]) / vs[1] / osf
                center_int = torch.stack([cx, cy]).to(torch.int32)
                draw_heatmap_gaussian(heatmap[labels[i]], center_int[[1, 0]],
                                      radius)
        return heatmap[None]

    # ------------------------------------------------------------ losses
    def loss_by_feat(self, preds_dicts, batch_gt_instances_3d, *args,
                     **kwargs):
        loss_dict = super().loss_by_feat(preds_dicts, batch_gt_instances_3d,
                                         *args, **kwargs)
        (_, _, bbox_targets, bbox_weights, _, num_pos, _,
         _) = self._cached_targets
        self._cached_targets = None
        preds = preds_dicts[0][0]
        avg = max(float(num_pos), 1.0)
        tgt = bbox_targets.permute(0, 2, 1)  # [B, code, N]
        wgt = bbox_weights.permute(0, 2, 1)

        if self.kappa != 0 and self.loss_center_weight > 0:
            coder = self.bbox_coder
            pc = coder.anchor_to_center_cells(preds['center'].float(),
                                              preds['dim'].float(),
                                              preds['rot'].float())
            tc = coder.anchor_to_center_cells(tgt[:, 0:2], tgt[:, 3:6],
                                              tgt[:, 6:8])
            loss_dict['loss_center_dec'] = self.loss_center_weight * (
                (pc - tc).abs() * wgt[:, 0:2]).sum() / avg

        if self.ssr:
            if self.loss_height_raw_weight > 0:
                loss_dict['loss_height_raw'] = self.loss_height_raw_weight * (
                    (preds['height_raw'].float() - tgt[:, 2:3]).abs() *
                    wgt[:, 2:3]).sum() / avg
            loss_dict['loss_plane'] = self.loss_plane_weight * \
                self._plane_loss(preds['plane'], batch_gt_instances_3d)
            if self.sea_field:
                # Laplace NLL of each matched query's own bottom estimate:
                # |e| / b + log b, with the error detached so that it only
                # calibrates the scale b (the heights keep their L1 losses)
                h_p = preds['dim'][:, 2:3].float().clamp(
                    *LOG_DIM_RANGE).exp()
                bot_p = preds['height_raw'].float() - 0.5 * h_p
                bot_t = tgt[:, 2:3] - 0.5 * tgt[:, 5:6].exp()
                err = (bot_p - bot_t).abs().detach()
                log_b = preds['sig'].float().clamp(*self.LOG_SIG_RANGE)
                nll = err * (-log_b).exp() + log_b
                loss_dict['loss_sigma'] = self.loss_sigma_weight * (
                    nll * (wgt[:, 2:3] > 0)).sum() / avg
        return loss_dict

    def _plane_loss(self, planes, batch_gt_instances_3d):
        """L1 between predicted planes [B, 3, L] and robust GT-bottom fits.

        Box tensors are bottom-centred (MaritimeDataset uses origin
        (0.5, 0.5, 0)), and they already carry the pipeline's
        augmentation, so the target plane lives in the same frame as the
        input. With ``imu_tilt`` only c is supervised, against the GT sea
        level under the same tilt (``_gt_sea``, cached by ``loss``).
        """
        losses = []
        if self.imu_tilt:
            for i, target in enumerate(self._train_planes or []):
                if target is not None:
                    losses.append((planes[i, 2] - target.to(planes)).abs()
                                  .mean())
            if not losses:
                return planes.sum() * 0.0
            return torch.stack(losses).mean()
        for i, gt in enumerate(batch_gt_instances_3d):
            boxes = gt.bboxes_3d.tensor
            if boxes.shape[0] < 3:
                continue
            target = fit_plane_torch(boxes[:, :3].to(planes.device).float(),
                                     self.plane_scale)
            if target is None:
                continue
            losses.append((planes[i] - target[:, None]).abs().sum(0).mean())
        if not losses:
            return planes.sum() * 0.0
        return torch.stack(losses).mean()
