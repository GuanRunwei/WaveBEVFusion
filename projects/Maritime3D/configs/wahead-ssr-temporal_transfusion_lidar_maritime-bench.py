"""SSR only (kappa=0) + temporal SSR: the sea plane is a recursive weighted
least-squares fit over each sequence (forgetting 0.9 per 0.1 s, reset after
a 1 s gap). Training simulates the history as a noisy pseudo observation of
the GT plane; evaluation walks every sequence in time order, one contiguous
chunk per GPU (SequentialChunkSampler).
The final-height loss is decoupled from the per-query raw heights and sizes
(ssr_decouple): with it coupled, bench_tfl_ssr at ep5 shrank h to 0.82x and
lifted the bottoms by +0.75 m (mAP3D 15.5 vs 21.0 for the baseline).
The temporal stream is test-time only, so evaluating this checkpoint with
temporal=False gives the per-frame number from the same weights.
"""
_base_ = ['./wahead-ssr-only_transfusion_lidar_maritime-bench.py']

model = dict(bbox_head=dict(temporal=True, ssr_decouple=True))

stream = dict(sampler=dict(_delete_=True, type='SequentialChunkSampler'))
val_dataloader = stream
test_dataloader = stream
