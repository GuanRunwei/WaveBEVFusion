"""Per-frame gravity direction for the IMU-anchored sea surface."""
import os
import warnings

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet3d.datasets.transforms.formating import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS

# Pack3DDetInputs' default meta keys plus the ones LoadSeaUp adds
_DEFAULT_META = Pack3DDetInputs.__init__.__defaults__[0]
SEA_UP_META_KEYS = tuple(_DEFAULT_META) + ('sea_up', 'sea_up_valid')
SEA_HEAVE_META_KEYS = SEA_UP_META_KEYS + ('sea_heave', 'sea_heave_valid',
                                          'sea_up_rot')


@TRANSFORMS.register_module()
class LoadSeaUp(BaseTransform):
    """Gravity up, in the (augmented) LiDAR frame, from a per-frame lookup.

    The LiDAR is body-fixed, so the sea tilts in its frame with the hull's
    attitude; the lookup holds the IMU-derived up vector for every frame,
    keyed by sequence and LiDAR timestamp (parsed from
    ``points4/<seq>/<ns>.bin``). Adds ``sea_up`` (float32 [3], unit) and
    ``sea_up_valid`` (bool); frames missing from the lookup, or flagged
    invalid, get (0, 0, 1) and ``sea_up_valid=False``. Pack them with
    ``Pack3DDetInputs(meta_keys=SEA_UP_META_KEYS)``.

    Place it after the augmentation: it applies ``lidar_aug_matrix`` (kept by
    the BEVFusion rotate / scale / translate / flip transforms), so the up
    vector follows the points. Other augmentations that do not record that
    matrix are refused rather than silently ignored.

    Args:
        lookup (str): npz with ``seq`` (str), ``ts`` (int64 ns), the up
            vectors (float [N, 3]) and ``valid`` (bool [N]).
        up_key (str): Which up vectors to use. dataset/infos/detection/
            sea_up_lidar.npz holds ``up_causal`` (past-only complementary
            filter, 5 s -- what an online system has, the default) and ``up``
            (zero-phase over +-10 s, an offline upper bound), both with the
            per-sequence mounting offset fitted on train+val horizons.
        heave_lookup (str, optional): npz with ``seq``, ``ts``, ``heave``
            (m, positive up, causal) and ``heave_valid``; adds ``sea_heave``
            and ``sea_heave_valid`` (pack with SEA_HEAVE_META_KEYS). Used by
            the sea-level Kalman prediction at test time only; heave is a
            relative displacement, so it is not touched by the augmentation.
        Always adds ``sea_up_rot`` (float32 [2, 2]): the map of the tilt from
        the body frame to the augmented frame (identity without
        augmentation), for the per-axis tilt gain of the sea surface.
    """

    def __init__(self, lookup: str, up_key: str = 'up_causal',
                 heave_lookup: str = None) -> None:
        self.lookup = lookup
        self.up_key = up_key
        self.heave_lookup = heave_lookup
        self._table = None  # loaded lazily, once per worker
        self._heave = None

    def _load(self):
        d = np.load(self.lookup, allow_pickle=False)
        up = d[self.up_key].astype(np.float64)
        valid = d['valid'].astype(bool)
        self._table = {(str(s), int(t)): (up[i], bool(valid[i]))
                       for i, (s, t) in enumerate(zip(d['seq'], d['ts']))}
        self._heave = {}
        if self.heave_lookup and not os.path.exists(self.heave_lookup):
            warnings.warn(f'{self.heave_lookup} not found: every frame gets '
                          'sea_heave_valid=False')
        elif self.heave_lookup:
            hv = np.load(self.heave_lookup, allow_pickle=False)
            self._heave = {
                (str(s), int(t)): (float(h), bool(v))
                for s, t, h, v in zip(hv['seq'], hv['ts'], hv['heave'],
                                      hv['heave_valid'])
            }

    def transform(self, results: dict) -> dict:
        if self._table is None:
            self._load()
        parts = str(results.get('lidar_path', '')).replace('\\',
                                                            '/').split('/')
        try:
            key = (parts[-2], int(parts[-1].split('.')[0]))
        except (IndexError, ValueError):
            key = None
        up, valid = self._table.get(key, (None, False))
        if up is None:
            up, valid = np.array([0.0, 0.0, 1.0]), False
        aug = results.get('lidar_aug_matrix')
        rot = np.eye(2)
        if aug is not None:
            m = np.asarray(aug, dtype=np.float64)[:3, :3]
            up = m @ up
            # tilt (a, b) = -s (up_x, up_y) / up_z maps from the body to the
            # augmented frame by m[:2, :2] / m[2, 2] (z rotation, BEV flips,
            # uniform scale): the frame in which a per-axis (pitch / roll)
            # correction has to be applied
            rot = m[:2, :2] / m[2, 2]
        elif any(k in results for k in ('pcd_rotation', 'pcd_horizontal_flip',
                                        'pcd_vertical_flip')):
            raise RuntimeError(
                'LoadSeaUp needs lidar_aug_matrix (BEVFusion augmentation '
                'transforms) to follow the augmentation of the points')
        results['sea_up'] = (up / np.linalg.norm(up)).astype(np.float32)
        results['sea_up_valid'] = bool(valid)
        results['sea_up_rot'] = rot.astype(np.float32)
        heave, hvalid = self._heave.get(key, (0.0, False))
        results['sea_heave'] = float(heave)
        results['sea_heave_valid'] = bool(hvalid)
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(lookup={self.lookup!r}, '
                f'up_key={self.up_key!r}, '
                f'heave_lookup={self.heave_lookup!r})')
