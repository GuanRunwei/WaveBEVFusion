"""gn_centerpoint-ea-sea_maritime-bench.py trained for 80 epochs (final
model; the ablations use 20). The schedule is re-declared because its end
points are fixed when the base config is parsed.
"""
_base_ = ['./gn_centerpoint-ea-sea_maritime-bench.py']

epoch_num = 80
train_cfg = dict(by_epoch=True, max_epochs=epoch_num, val_interval=5)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.33333333, by_epoch=False, begin=0,
         end=500),
    dict(type='CosineAnnealingLR', begin=0, T_max=epoch_num, end=epoch_num,
         by_epoch=True, eta_min_ratio=1e-4, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=0.8947, begin=0,
         end=epoch_num * 0.4, by_epoch=True, convert_to_iter_based=True),
    dict(type='CosineAnnealingMomentum', eta_min=1, begin=epoch_num * 0.4,
         end=epoch_num, by_epoch=True, convert_to_iter_based=True),
]
