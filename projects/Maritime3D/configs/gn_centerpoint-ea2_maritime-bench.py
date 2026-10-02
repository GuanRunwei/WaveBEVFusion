"""CenterPoint (GroupNorm, 20 epochs) with evidence anchors, local-frame
anchor-to-centre vector.

Same as gn_centerpoint-ea_maritime-bench.py (heatmap peak at the centroid of
each hull's LiDAR returns) but the anchor-to-centre vector is regressed in
the box frame as fractions of the hull, (alpha along the length, beta
across the width), instead of metres in the BEV frame: bounded (|.| <= 0.5
plus the margin), dimensionless, the same range for a 6 m boat and a 70 m
ship, and decoded with the predicted length / width / yaw. The BEV-frame
vector (gn_cp_ea) recovered the objects (AP@20m up) but not their centres
(boat AP@2m 57 -> 39, ship 24 -> 4).
"""
_base_ = ['./gn_centerpoint-ea_maritime-bench.py']

model = dict(
    pts_bbox_head=dict(
        a2c_scale=0.5, a2c_frame='local',
        bbox_coder=dict(a2c_scale=0.5, a2c_frame='local')))
