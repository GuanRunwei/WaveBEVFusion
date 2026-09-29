"""Physical sea-surface model, shared by the maritime detection heads.

The sea height under a point p of the (augmented) LiDAR frame is

    s(p) = tilt(p) + c + eta(p),

* tilt(p) = (a x + b y) / s: the mean sea is normal to gravity, so its tilt
  comes from the IMU (``sea_up`` meta, see LoadSeaUp), scaled per body axis
  (pitch, roll) by a learned gain -- the annotated heights follow the hull's
  attitude only partly -- plus a learned mounting offset;
* c ~ N(c-, P-): the mean level, one Kalman step per frame. The prior is
  simulated in training (the GT level plus noise of random std s, P- = s^2)
  and streamed at test time (random walk with process noise ``kalman_q``
  per 0.1 s; IMU heave, measured to carry <1% of the annotated level's
  change, is optional via ``heave_gain``);
* eta: a zero-mean Gaussian-process wave field, k(p, q) = sw^2 l^2 / lij^2
  exp(-|p-q|^2 / 2 lij^2), lij^2 = l^2 + (L_p^2 + L_q^2) / 12: waves
  averaged over each hull's length L, so long ships ride over them. sw and l
  are learned. Box bottoms deviate from a frame's shared level by 0.36 m
  median and the deviations of two hulls correlate +0.52 within 20 m and ~0
  beyond 50 m (train split): local waves, not one rigid plane.

Detections observe their own waterline, y_j = s(p_j) + e_j with
e_j ~ N(0, 2 b_j^2 / score_j), b_j a learned Laplace scale. One closed-form
solve gives the Kalman update of c and the kriged sea height at every query:
a well-observed hull keeps its own waterline, a sparse far one borrows from
neighbours within ~l, an isolated one falls back to the mean sea. sw -> 0
recovers a rigid plane; no observations recover the Kalman prior.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch import Tensor

LOG_SIG_RANGE = (-3.0, 2.0)  # Laplace scale of a waterline: 5 cm .. 7.4 m


class SeaSurface(nn.Module):
    """IMU tilt + Kalman mean level + GP wave field (see module docstring).

    Args:
        plane_scale (float): Metres per unit of the tilt terms a, b.
        prior_c (float): Initial no-history mean level (LiDAR frame, m).
        level_prior_var (float): Initial no-history variance of c (m^2).
        wave_sigma (float): Initial wave amplitude sw (m).
        wave_length (float): Initial wave correlation length l (m).
        tilt_gain (bool): Learn a per-body-axis gain on the IMU tilt; needs
            the ``sea_up_rot`` meta in training.
        max_obs (int): Most confident detections used as observations.
        hist_prob (float): Training: probability of a simulated history.
        hist_std_range (tuple): Training: range of the simulated history std.
        kalman_q (float): Process noise of c per 0.1 s (m^2).
        heave_gain (float): Test time: c- = c_prev - gain * d(heave).
        max_gap (float): Test time: longer gaps (s) reset the stream.
    """

    def __init__(self,
                 plane_scale: float = 100.0,
                 prior_c: float = -1.3,
                 level_prior_var: float = 1.0,
                 wave_sigma: float = 0.3,
                 wave_length: float = 25.0,
                 tilt_gain: bool = True,
                 max_obs: int = 50,
                 hist_prob: float = 0.5,
                 hist_std_range: Sequence[float] = (0.05, 1.0),
                 kalman_q: float = 0.005,
                 heave_gain: float = 0.0,
                 max_gap: float = 1.0) -> None:
        super().__init__()
        self.scale = float(plane_scale)
        self.max_obs = int(max_obs)
        self.hist_prob = float(hist_prob)
        self.hist_std_range = tuple(float(v) for v in hist_std_range)
        self.kalman_q = float(kalman_q)
        self.heave_gain = float(heave_gain)
        self.max_gap = float(max_gap)
        self.prior_c = nn.Parameter(torch.tensor(float(prior_c)))
        self.prior_ab = nn.Parameter(torch.zeros(2))  # frames without IMU
        self.tilt_offset = nn.Parameter(torch.zeros(2))
        # gain = 1 + delta, so weight decay pulls towards the IMU tilt
        self.tilt_gain_delta = nn.Parameter(torch.zeros(2)) \
            if tilt_gain else None
        self.log_level_prior_var = nn.Parameter(
            torch.tensor(float(level_prior_var)).log())
        self.log_wave_sigma = nn.Parameter(
            torch.tensor(float(wave_sigma)).log())
        self.log_wave_length = nn.Parameter(
            torch.tensor(float(wave_length)).log())
        self._stream = {}  # seq -> (timestamp_ns, c, P, heave or None)

    # ------------------------------------------------------------ helpers
    def _keep(self) -> Tensor:
        """0 * every parameter: keeps them all in the graph (DDP)."""
        return sum(0 * p.sum() for p in self.parameters())

    @staticmethod
    def stream_key(meta: dict):
        """(seq, timestamp in ns) from points4/<seq>/<ns>.bin, or None."""
        parts = str(meta.get('lidar_path', '')).replace('\\', '/').split('/')
        try:
            return parts[-2], int(parts[-1].split('.')[0])
        except (IndexError, ValueError):
            return None

    def reset_stream(self) -> None:
        self._stream = {}

    def tilt(self, metas: List[dict], device) -> Tensor:
        """Mean-sea tilt (a, b) [B, 2] in the augmented LiDAR frame."""
        s = self.scale
        gain = None if self.tilt_gain_delta is None \
            else 1 + self.tilt_gain_delta
        rows = []
        for meta in metas:
            up = meta.get('sea_up')
            if up is None or not bool(meta.get('sea_up_valid', False)):
                rows.append(self.prior_ab)
                continue
            up = torch.as_tensor(up, dtype=torch.float32, device=device)
            ab = -s * up[:2] / up[2]
            if gain is not None:
                rot = meta.get('sea_up_rot')
                if rot is None:
                    assert not self.training, \
                        'the tilt gain needs the sea_up_rot meta in training'
                    rot = torch.eye(2)
                rot = torch.as_tensor(rot, dtype=torch.float32, device=device)
                # body frame -> scale pitch / roll -> augmented frame
                ab = rot @ (gain * torch.linalg.solve(rot, ab))
            rows.append(ab + self.tilt_offset)
        return torch.stack(rows) + self._keep()

    def tilt_z(self, tilt: Tensor, x: Tensor, y: Tensor) -> Tensor:
        """Tilt term (a x + b y) / s for x, y [B, n]."""
        return (tilt[:, 0:1] * x + tilt[:, 1:2] * y) / self.scale

    def gt_level(self, xyz_bottom: Tensor, tilt: Tensor) -> Optional[Tensor]:
        """GT mean level of one frame: median of the box bottoms after
        removing the (detached) tilt; None without boxes."""
        if xyz_bottom.shape[0] == 0:
            return None
        t = tilt.detach().to(xyz_bottom)
        level = xyz_bottom[:, 2] - (t[0] * xyz_bottom[:, 0] +
                                    t[1] * xyz_bottom[:, 1]) / self.scale
        return level.median()

    # ------------------------------------------------------- Kalman prior
    def level_prior(self, metas: List[dict],
                    gt_levels: Optional[List[Optional[Tensor]]] = None):
        """One-step Kalman prior (c-, P-) of the mean level, [B] each."""
        c0 = self.prior_c
        P0 = self.log_level_prior_var.exp()
        lo, hi = self.hist_std_range
        c, P = [], []
        for i, meta in enumerate(metas):
            ci, Pi = c0, P0
            if self.training:
                gt = None if gt_levels is None else gt_levels[i]
                if gt is not None and torch.rand(()) < self.hist_prob:
                    s = lo + (hi - lo) * float(torch.rand(()))
                    ci = gt.detach() + s * torch.randn((), device=c0.device)
                    Pi = c0.new_tensor(s * s)
            else:
                key = self.stream_key(meta)
                prev = self._stream.get(key[0]) if key else None
                if prev is not None:
                    ts, cp, Pp, hp = prev
                    dt = (key[1] - ts) * 1e-9
                    if 0 < dt <= self.max_gap:
                        h = meta.get('sea_heave')
                        dh = 0.0
                        if hp is not None and h is not None and \
                                bool(meta.get('sea_heave_valid', False)):
                            dh = float(h) - hp
                        ci = c0.new_tensor(cp - self.heave_gain * dh)
                        Pi = c0.new_tensor(Pp + self.kalman_q * dt / 0.1)
            c.append(ci)
            P.append(Pi)
        keep = self._keep()
        return torch.stack(c) + keep, torch.stack(P) + keep

    def update_stream(self, metas: List[dict], c: Tensor, P: Tensor) -> None:
        """Store this frame's posterior level (test time)."""
        for i, meta in enumerate(metas):
            key = self.stream_key(meta)
            if key is None:
                continue
            h = meta.get('sea_heave')
            ok = h is not None and bool(meta.get('sea_heave_valid', False))
            self._stream[key[0]] = (key[1], float(c[i]), float(P[i]),
                                    float(h) if ok else None)

    # ------------------------------------------------------ Kalman + GP
    def wave_cov(self, xa, ya, La, xb, yb, Lb) -> Tensor:
        """Wave covariance [B, na, nb] between hull-averaged sea heights."""
        sw2 = (2 * self.log_wave_sigma).exp()
        l2 = (2 * self.log_wave_length).exp()
        lij2 = l2 + (La[:, :, None]**2 + Lb[:, None, :]**2) / 12.0
        d2 = (xa[:, :, None] - xb[:, None, :])**2 + \
            (ya[:, :, None] - yb[:, None, :])**2
        return sw2 * (l2 / lij2) * torch.exp(-0.5 * d2 / lij2)

    def solve(self, tilt: Tensor, c_prior: Tensor, P_prior: Tensor,
              obs: Dict[str, Tensor], qry: Dict[str, Tensor]):
        """Sea height at the queries and the posterior mean level.

        obs: x, y, L (hull length), bottom (observed waterline), log_b,
            score and mask, [B, M] each; the positions, lengths and bottoms
            should be detached (only b, the tilt and the sea parameters
            learn through the fused height).
        qry: x, y, L, [B, Q] each.
        Returns sea [B, Q], c_post [B], P_post [B].
        """
        m = obs['mask'].float()
        if m.shape[1] > self.max_obs:  # the most confident observations
            keep = (obs['score'] * m).topk(self.max_obs, dim=1).indices
            obs = {k: v.gather(1, keep) for k, v in obs.items()}
            m = obs['mask'].float()
        x, y, L = obs['x'], obs['y'], obs['L']
        innov = (obs['bottom'] - self.tilt_z(tilt, x, y) -
                 c_prior[:, None]) * m
        log_b = obs['log_b'].clamp(*LOG_SIG_RANGE)
        R = 2 * (2 * log_b).exp() / obs['score'].clamp_min(1e-3)
        R = torch.where(m > 0, R, torch.full_like(R, 1e6))
        mm = m[:, :, None] * m[:, None, :]
        Pm = P_prior[:, None, None]
        S = Pm * mm + self.wave_cov(x, y, L, x, y, L) * mm + \
            torch.diag_embed(R)
        S = S + 1e-4 * torch.eye(S.shape[-1], device=S.device)
        rhs = torch.stack([innov, m], dim=-1)
        sol = torch.linalg.solve(S.double(), rhs.double()).to(S.dtype)
        alpha, s_inv_h = sol[..., 0], sol[..., 1]              # [B, M]
        c_post = c_prior + P_prior * (m * alpha).sum(1)
        P_post = P_prior - P_prior**2 * (m * s_inv_h).sum(1)
        # kriging: cov(s(q), y_O) = P- + k_wave(q, O), masked columns zero
        Kx = (Pm + self.wave_cov(qry['x'], qry['y'], qry['L'], x, y, L)) * \
            m[:, None, :]
        sea = self.tilt_z(tilt, qry['x'], qry['y']) + c_prior[:, None] + \
            (Kx @ alpha[..., None])[..., 0]
        return sea, c_post, P_post
