"""Full SSR with 2 decoder layers: decoupled, temporal, IMU tilt, and the
sea context -- the plane solved after layer 1 is embedded per query (own
bottom minus sea height, a, b, c) and added to the query features entering
layer 2. Compare with transfusion_lidar-2dec_maritime-bench.py.
"""
_base_ = ['./wahead-ssr-imu-temporal_transfusion_lidar_maritime-bench.py']

model = dict(
    bbox_head=dict(num_decoder_layers=2, sea_context=True))
