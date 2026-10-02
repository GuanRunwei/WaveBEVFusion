"""Ablation: gn_centerpoint-sea_maritime-bench-80e.py with the sea surface also applied at test time
(sea_at_test=True): the detections' bottoms are fused with the sea surface
(Kalman level + GP waves) instead of keeping the head's own heights. Same
checkpoint as gn_centerpoint-sea_maritime-bench-80e.py; evaluation only.
"""
_base_ = ['./gn_centerpoint-sea_maritime-bench-80e.py']

model = dict(pts_bbox_head=dict(sea_at_test=True))
