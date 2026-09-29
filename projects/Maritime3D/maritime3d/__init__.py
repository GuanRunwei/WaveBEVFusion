"""Maritime3D dataset, evaluation metric and shared config pieces.

Registered under mmdet3d's registries so that configs can refer to
`MaritimeDataset` / `MaritimeMetric` by name.
"""
from .evidence_centerpoint import (EvidenceAnchor,
                                   EvidenceCenterPointBBoxCoder,
                                   EvidenceSeaCenterHead,
                                   Pack3DDetInputsAnchor)
from .maritime_dataset import MaritimeDataset
from .maritime_metric import MaritimeMetric
from .match_cost import NormalizedBBox3DL1Cost
from .pose_boxes import MaritimePoseBoxes
from .pose_centerpoint import (MaritimeCenterHead,
                               MaritimeClassBalancedGaussianFocalLoss,
                               MaritimePoseCenterPointBBoxCoder)
from .samplers import SequentialChunkSampler
from .sea_surface import SeaSurface
from .sea_up import SEA_HEAVE_META_KEYS, SEA_UP_META_KEYS, LoadSeaUp
from .transforms import (MaritimeGlobalRotScaleTransImage,
                         ObjectAzimuthFilter)

__all__ = [
    'EvidenceAnchor', 'EvidenceCenterPointBBoxCoder', 'EvidenceSeaCenterHead',
    'Pack3DDetInputsAnchor', 'SeaSurface',
    'SequentialChunkSampler', 'LoadSeaUp', 'SEA_UP_META_KEYS', 'SEA_HEAVE_META_KEYS',
    'MaritimeDataset', 'MaritimeMetric', 'ObjectAzimuthFilter',
    'MaritimeGlobalRotScaleTransImage', 'NormalizedBBox3DL1Cost',
    'MaritimePoseBoxes', 'MaritimeCenterHead',
    'MaritimeClassBalancedGaussianFocalLoss',
    'MaritimePoseCenterPointBBoxCoder'
]
