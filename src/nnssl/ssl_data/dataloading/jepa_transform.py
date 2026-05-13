"""
batchgenerators-compatible JEPA mask transform.

Wraps :class:`PatchMaskSampler3D` as an ``AbstractTransform`` so it can be
appended to the standard nnssl augmentation pipeline.  Adds ``context_mask``
and ``target_masks`` keys to the data dict as numpy bool arrays (converted to
torch tensors in ``train_step``).
"""

from __future__ import annotations

import torch
import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform

from nnssl.ssl_data.dataloading.jepa_masking import PatchMaskSampler3D, MaskSamplerConfig


class JEPAMaskTransform(AbstractTransform):
    """Adds JEPA context/target masks to the batchgenerators data dict.

    Must be placed **after** ``NumpyToTensor`` in the training transform chain
    so that ``data_dict["data"]`` has the final shape.

    Keys added to ``data_dict``:
        ``context_mask``  — ``np.ndarray`` bool ``[B, N]``
        ``target_masks``  — ``np.ndarray`` bool ``[B, T, N]``
    """

    def __init__(
        self,
        patch_grid: tuple[int, int, int],
        num_target_blocks: int = 4,
        context_scale_range: tuple[float, float] = (0.85, 1.0),
        target_scale_range: tuple[float, float] = (0.15, 0.2),
        target_aspect_ratio: tuple[float, float] = (0.75, 1.5),
    ):
        cfg = MaskSamplerConfig(
            patch_grid=patch_grid,
            context_scale_range=context_scale_range,
            target_scale_range=target_scale_range,
            target_aspect_ratio=target_aspect_ratio,
            num_target_blocks=num_target_blocks,
        )
        self.sampler = PatchMaskSampler3D(cfg)

    def __call__(self, **data_dict):
        # data_dict["data"] may be a torch Tensor or numpy array at this point
        data = data_dict["data"]
        if isinstance(data, torch.Tensor):
            B = data.shape[0]
        else:
            B = data.shape[0]

        # Sample on CPU as numpy bools (moved to GPU in train_step)
        ctx_mask, tgt_masks = self.sampler(B, device=torch.device("cpu"))
        data_dict["context_mask"] = ctx_mask.numpy()
        data_dict["target_masks"] = tgt_masks.numpy()
        return data_dict
