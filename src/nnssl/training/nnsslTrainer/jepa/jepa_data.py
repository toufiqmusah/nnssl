"""
Data pipeline components for JEPA 3D medical image self-supervised pretraining.

Implements:
- BlockMaskGenerator3D: Generates 3D block masks at the patch level.
- JEPATransform: batchgenerators-compatible transform for I-JEPA (ViT).
- JEPA3DTransform: batchgenerators-compatible transform for 3D-JEPA (CNN).

Masking philosophy:
  Masking operates at the *patch* level (spatially contiguous blocks of
  voxels), not individual voxels.  This encourages the model to learn
  semantically meaningful representations rather than trivial interpolation.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor

from nnssl.training.nnsslTrainer.jepa.jepa_config import IJEPAConfig, JEPA3DConfig, MaskingConfig


class BlockMaskGenerator3D:
    """Generates 3D block masks at patch resolution.

    Masks are composed of K spatially-contiguous rectangular blocks of patches.
    This ensures that the model must predict entire semantic regions rather than
    isolated voxels, driving more useful representations.

    Args:
        grid_size: Number of patches along each axis (gX, gY, gZ).
        masking_config: MaskingConfig dataclass.
        seed: Optional random seed for reproducibility.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        masking_config: MaskingConfig,
        seed: Optional[int] = None,
    ) -> None:
        self.grid_size = grid_size
        self.num_patches = grid_size[0] * grid_size[1] * grid_size[2]
        self.num_target_blocks = masking_config.num_target_blocks
        self.target_block_size = masking_config.target_block_size
        self.context_fraction = masking_config.context_fraction
        self._rng = np.random.default_rng(seed)

    def _sample_block(self, block_size: Tuple[int, int, int]) -> List[int]:
        """Sample a random rectangular block of patch indices.

        Args:
            block_size: Size of the block in patch units (dx, dy, dz).
                Clipped to grid boundaries automatically.

        Returns:
            Sorted list of 1-D patch indices within the block.
        """
        gx, gy, gz = self.grid_size
        bx, by, bz = block_size

        # Clip block size to not exceed grid
        bx = min(bx, gx)
        by = min(by, gy)
        bz = min(bz, gz)

        # Sample top-left corner of the block uniformly
        x0 = int(self._rng.integers(0, gx - bx + 1))
        y0 = int(self._rng.integers(0, gy - by + 1))
        z0 = int(self._rng.integers(0, gz - bz + 1))

        indices = []
        for ix in range(x0, x0 + bx):
            for iy in range(y0, y0 + by):
                for iz in range(z0, z0 + bz):
                    idx = ix * gy * gz + iy * gz + iz
                    indices.append(idx)
        return sorted(set(indices))

    def __call__(self, batch_size: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate context and target masks for a batch.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Tuple of:
                - context_indices: Long tensor (B, n_ctx) — patch indices
                  visible to the context encoder.  Shape varies since
                  different samples may have different context sizes; in
                  practice context regions are kept the same size per-sample
                  by construction.
                - target_indices_list: List of K long tensors (B, n_tgt_k)
                  — patch indices for each of the K target blocks.
        """
        all_context: List[List[int]] = []
        per_block_targets: List[List[List[int]]] = [[] for _ in range(self.num_target_blocks)]

        for _ in range(batch_size):
            # Sample K target blocks for this sample
            target_set: set[int] = set()
            target_blocks: List[List[int]] = []
            for k in range(self.num_target_blocks):
                block_indices = self._sample_block(self.target_block_size)
                target_blocks.append(block_indices)
                target_set.update(block_indices)

            # Context: all patches NOT in any target block
            context_all = [i for i in range(self.num_patches) if i not in target_set]
            # Optionally subsample context
            if self.context_fraction < 1.0:
                n_ctx = max(1, int(len(context_all) * self.context_fraction))
                ctx_perm = self._rng.permutation(len(context_all))[:n_ctx]
                context_all = sorted([context_all[j] for j in ctx_perm])

            all_context.append(context_all)
            for k in range(self.num_target_blocks):
                per_block_targets[k].append(target_blocks[k])

        # Pad context sequences to the same length across the batch
        min_ctx_len = min(len(c) for c in all_context)
        # Trim to minimum length for batch-consistent tensor creation
        context_indices = torch.tensor(
            [c[:min_ctx_len] for c in all_context], dtype=torch.long
        )  # (B, n_ctx)

        # Build target index tensors (one per block)
        target_indices_list: List[torch.Tensor] = []
        for k in range(self.num_target_blocks):
            min_tgt_len = min(len(b) for b in per_block_targets[k])
            tgt_k = torch.tensor(
                [b[:min_tgt_len] for b in per_block_targets[k]], dtype=torch.long
            )  # (B, n_tgt_k)
            target_indices_list.append(tgt_k)

        return context_indices, target_indices_list


class JEPATransform(AbstractTransform):
    """batchgenerators-compatible data transform for I-JEPA.

    Generates block masks at the patch level and stores them in the
    data dictionary alongside the raw image data.  No heavy augmentation
    is applied here—spatial and intensity augmentations are applied
    upstream by the standard nnssl pipeline.

    The transform adds the following keys to the data dict:
        ``"context_indices"`` — LongTensor (B, n_ctx)
        ``"target_indices_list"`` — list of K LongTensors (B, n_tgt_k)

    Args:
        config: IJEPAConfig, used to derive grid size and masking params.
    """

    def __init__(self, config: IJEPAConfig) -> None:
        self.config = config
        vol = config.volume_size
        ps = config.patch_size
        grid_size: Tuple[int, int, int] = (
            vol[0] // ps[0],
            vol[1] // ps[1],
            vol[2] // ps[2],
        )
        self.mask_generator = BlockMaskGenerator3D(
            grid_size=grid_size,
            masking_config=config.masking,
        )

    def __call__(self, **data_dict) -> dict:
        """Apply the transform.

        Expects ``data_dict["data"]`` to have shape ``(B, C, X, Y, Z)``.
        """
        data = data_dict.get("data")
        if data is None:
            raise KeyError('"data" key is required in data_dict for JEPATransform.')

        batch_size = data.shape[0]
        context_indices, target_indices_list = self.mask_generator(batch_size)

        data_dict["context_indices"] = context_indices
        data_dict["target_indices_list"] = target_indices_list
        return data_dict


class JEPA3DTransform(AbstractTransform):
    """batchgenerators-compatible data transform for 3D-JEPA.

    Functionally identical to JEPATransform but uses JEPA3DConfig to
    derive the grid parameters, supporting larger volumes and coarser
    patch hierarchies suited to volumetric 3D medical imaging.

    Args:
        config: JEPA3DConfig, used to derive grid size and masking params.
    """

    def __init__(self, config: JEPA3DConfig) -> None:
        self.config = config
        vol = config.volume_size
        ps = config.patch_size
        grid_size: Tuple[int, int, int] = (
            vol[0] // ps[0],
            vol[1] // ps[1],
            vol[2] // ps[2],
        )
        self.mask_generator = BlockMaskGenerator3D(
            grid_size=grid_size,
            masking_config=config.masking,
        )

    def __call__(self, **data_dict) -> dict:
        """Apply the transform.

        Expects ``data_dict["data"]`` to have shape ``(B, C, X, Y, Z)``.
        """
        data = data_dict.get("data")
        if data is None:
            raise KeyError('"data" key is required in data_dict for JEPA3DTransform.')

        batch_size = data.shape[0]
        context_indices, target_indices_list = self.mask_generator(batch_size)

        data_dict["context_indices"] = context_indices
        data_dict["target_indices_list"] = target_indices_list
        return data_dict


# ---------------------------------------------------------------------------
# Multi-scale hierarchical masking (optional extension)
# ---------------------------------------------------------------------------


class MultiScaleBlockMaskGenerator3D:
    """Multi-scale block masking hierarchy for 3D volumes.

    Combines multiple BlockMaskGenerator3D instances at different scales
    (fine, medium, coarse) to encourage learning at multiple semantic levels.

    Args:
        grid_size: Number of patches along each axis (gX, gY, gZ).
        masking_configs: List of MaskingConfig objects, one per scale.
            Scales are applied in order from fine-to-coarse.
    """

    def __init__(
        self,
        grid_size: Tuple[int, int, int],
        masking_configs: List[MaskingConfig],
    ) -> None:
        self.generators = [
            BlockMaskGenerator3D(grid_size=grid_size, masking_config=cfg)
            for cfg in masking_configs
        ]

    def __call__(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Generate multi-scale masks.

        Aggregates target blocks across all scales; context is the
        complement of all target blocks.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Same interface as BlockMaskGenerator3D.
        """
        # Generate masks from the finest-scale generator as primary reference
        context_indices, target_indices_list = self.generators[0](batch_size)

        # Add target blocks from coarser-scale generators
        for gen in self.generators[1:]:
            _, extra_targets = gen(batch_size)
            target_indices_list.extend(extra_targets)

        return context_indices, target_indices_list
