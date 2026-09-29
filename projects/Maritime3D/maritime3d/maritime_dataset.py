# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union

import numpy as np

from mmdet3d.datasets.det3d_dataset import Det3DDataset
from mmdet3d.registry import DATASETS
from .pose_boxes import MaritimePoseBoxes


@DATASETS.register_module()
class MaritimeDataset(Det3DDataset):
    """Maritime 3D detection dataset (sequence 00 of the maritime3d capture).

    Annotations are stored directly in the LiDAR frame, so unlike KITTI there
    is no rectification step: `bbox_3d` is already
    [x, y, z_bottom, dx, dy, dz, yaw] in LiDAR coordinates.

    Args:
        data_root (str): Dataset root.
        ann_file (str): Annotation pkl relative to ``data_root``.
        pipeline (list): Processing pipeline.
        modality (dict): Which sensors to load. Defaults to LiDAR only.
        default_cam_key (str): Camera used by image-based branches.
        box_type_3d (str): Box coordinate system. Defaults to 'LiDAR'.
            'Maritime' gives LiDAR boxes whose columns 7/8 are pitch/roll
            (:class:`MaritimePoseBoxes`, use with ``box_3d_dof=9``); stock
            LiDAR boxes would rotate and scale those columns as velocity.
        filter_empty_gt (bool): Drop frames without GT. Evaluation splits
            should set this to False so that false positives on empty water
            are still counted.
        pcd_limit_range (list): Range used to filter predicted boxes.
    """

    METAINFO = {
        'classes': ('boat', 'ship', 'sailboat', 'buoy'),
        'palette': [(0, 120, 255), (255, 90, 0), (0, 200, 120), (255, 210, 0)],
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 modality: dict = dict(use_lidar=True, use_camera=False),
                 default_cam_key: str = 'CAM_LEFT',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [
                     -160.0, -160.0, -8.0, 160.0, 160.0, 24.0
                 ],
                 box_3d_dof: int = 7,
                 empty_frame_keep: float = 1.0,
                 **kwargs) -> None:
        self.pcd_limit_range = pcd_limit_range
        # Fraction of GT-free frames kept for training (57% of the train
        # split is open water). Subsampled deterministically in
        # filter_data; evaluation splits always keep every frame.
        assert 0.0 <= empty_frame_keep <= 1.0
        self.empty_frame_keep = empty_frame_keep
        # How many leading dims of the stored 11-wide bbox_3d
        # [x, y, z, dx, dy, dz, yaw, pitch, roll, pad, pad] to expose.
        # 7 (default) matches every standard 7-DoF head; 9 additionally
        # carries target pitch/roll for 10-DoF baselines. The remaining
        # per-instance fields (gt_pitch/gt_roll/...) stay in the pkl either
        # way.
        assert box_3d_dof in (7, 9), \
            'box_3d_dof must be 7 (standard) or 9 (+pitch/roll)'
        self.box_3d_dof = box_3d_dof
        # Det3DDataset derives label_mapping from self.METAINFO['classes']
        # and treats config metainfo as a SUBSET of it. The v5 pkls use a
        # 5-class vocabulary (adds 'yacht'), so when the configured classes
        # are not a subset of the legacy vocabulary, adopt the configured
        # vocabulary as the instance-level METAINFO: pkl labels are already
        # in that order and the mapping becomes the identity. Legacy 4-class
        # configs keep the old subset-mapping behaviour unchanged.
        kw_meta = kwargs.get('metainfo')
        if kw_meta is not None and 'classes' in kw_meta and \
                not set(kw_meta['classes']) <= set(self.METAINFO['classes']):
            self.METAINFO = dict(
                classes=tuple(kw_meta['classes']),
                palette=kw_meta.get(
                    'palette', MaritimeDataset.METAINFO['palette']))
        # get_box_type only knows the stock types, and annotations are
        # parsed inside Det3DDataset.__init__, so the pose box class has to
        # be known before that call
        pose = box_type_3d.lower() == 'maritime'
        if pose:
            assert box_3d_dof == 9, "box_type_3d='Maritime' needs box_3d_dof=9"
        self._box_cls = MaritimePoseBoxes if pose else None
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d='LiDAR' if pose else box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        if pose:
            # handed to the pipeline and to the heads via the data metas
            self.box_type_3d = MaritimePoseBoxes

    def filter_data(self) -> List[dict]:
        """Keep every GT frame and every k-th GT-free frame (training only).

        Det3DDataset leaves empty frames in ``data_list`` and, with
        ``filter_empty_gt=True``, resamples a random index when it hits one --
        wasteful when most frames are empty, and it silently changes the
        sampling distribution. Dropping them here instead keeps an explicit,
        reproducible share of true negatives.
        """
        if self.test_mode or self.empty_frame_keep >= 1.0:
            return self.data_list
        keep, n_empty = [], 0
        step = (0 if self.empty_frame_keep <= 0 else max(
            1, int(round(1.0 / self.empty_frame_keep))))
        for info in self.data_list:
            if info.get('instances'):
                keep.append(info)
                continue
            if step and n_empty % step == 0:
                keep.append(info)
            n_empty += 1
        return keep

    def parse_ann_info(self, info: dict) -> dict:
        """Fill in empty arrays so that frames without GT stay usable."""
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict(
                gt_bboxes_3d=np.zeros((0, self.box_3d_dof), dtype=np.float32),
                gt_labels_3d=np.zeros(0, dtype=np.int64))
        boxes = ann_info['gt_bboxes_3d']
        # The new_annotations pkls store 11-wide boxes
        # [x, y, z, dx, dy, dz, yaw, pitch, roll, pad, pad] ("10dof").
        # Expose the leading `box_3d_dof` columns: 7 for standard heads,
        # 9 to also supervise target pitch/roll (cols 7/8). The trailing
        # pads and the per-instance copies (gt_pitch/gt_roll/pitch_roll)
        # always stay in the pkl.
        if boxes.shape[-1] > self.box_3d_dof:
            boxes = boxes[:, :self.box_3d_dof]
        ann_info['gt_bboxes_3d'] = (self._box_cls or self.box_type_3d)(
            boxes,
            box_dim=boxes.shape[-1],
            origin=(0.5, 0.5, 0.0))
        return ann_info
