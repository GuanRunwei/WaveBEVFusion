"""TransFusion-L baseline with 2 decoder layers: the fair reference for the
sea-context SSR variant, which needs a second layer to feed the plane into.
Otherwise identical to transfusion_lidar_maritime-bench.py.
"""
_base_ = ['./transfusion_lidar_maritime-bench.py']

model = dict(bbox_head=dict(num_decoder_layers=2))
