"""Samplers for Maritime3D."""
from typing import Iterator

from mmengine.dist import get_dist_info
from torch.utils.data import Sampler

from mmdet3d.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SequentialChunkSampler(Sampler):
    """Evaluation sampler for stateful (streaming) models.

    Frames are sorted by (seq, timestamp) and every rank gets one contiguous
    chunk, so a model that carries state from frame to frame (temporal SSR)
    sees each sequence in time order; DefaultSampler would hand each rank
    every world_size-th frame instead.

    mmengine's collect_results re-interleaves the ranks' outputs and truncates
    to len(dataset). Chunks therefore differ in length by at most one and the
    shorter ones are padded at their end, so the padding lands exactly in the
    part that gets truncated.
    """

    def __init__(self, dataset, **kwargs) -> None:
        rank, world_size = get_dist_info()
        n = len(dataset)

        def key(i):
            info = dataset.get_data_info(i)
            return (str(info.get('seq', '')), int(info.get('timestamp', 0)), i)

        order = sorted(range(n), key=key)
        q, r = divmod(n, world_size)
        start = rank * q + min(rank, r)
        mine = order[start:start + q + (1 if rank < r else 0)]
        size = q + (1 if r else 0)
        if len(mine) < size:
            mine = mine + [mine[-1] if mine else order[-1]]
        self.indices = mine

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)

    def set_epoch(self, epoch: int) -> None:
        pass
