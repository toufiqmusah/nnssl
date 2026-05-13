"""
3D Block Mask Sampler for JEPA pre-training.

Generates complementary context and target masks for Joint Embedding Predictive
Architecture (JEPA) training.  Target blocks are sampled as random axis-aligned
3D boxes in the patch grid; the context mask is the complement of their union.

Reference: Assran et al., "Self-Supervised Learning from Images with a
Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023.
"""

from dataclasses import dataclass, field
from typing import Tuple

import torch
import numpy as np


@dataclass
class MaskSamplerConfig:
    """Configuration for :class:`PatchMaskSampler3D`."""

    patch_grid: Tuple[int, int, int]
    """Number of patches along each spatial axis (D, H, W)."""

    num_target_blocks: int = 4
    """Number of target blocks to sample per image."""

    context_scale_range: Tuple[float, float] = (0.85, 1.0)
    """Fraction of total patches visible to the context encoder."""

    target_scale_range: Tuple[float, float] = (0.15, 0.20)
    """Fraction of total patches covered by each target block."""

    target_aspect_ratio: Tuple[float, float] = (0.75, 1.5)
    """Log-uniform aspect-ratio range for target blocks."""


class PatchMaskSampler3D:
    """Samples 3D block masks for JEPA pre-training.

    For each sample in a batch, generates:
      * ``target_masks`` — ``T`` boolean masks, each covering a contiguous 3D
        block of the patch grid.
      * ``context_mask`` — the complement of the union of all target blocks
        (optionally further sub-sampled to the ``context_scale_range``).

    Shapes (after call):
      * ``context_mask``:  ``[B, N]``  bool
      * ``target_masks``:  ``[B, T, N]``  bool
      where ``N = prod(patch_grid)`` and ``T = num_target_blocks``.
    """

    def __init__(self, cfg: MaskSamplerConfig):
        self.cfg = cfg
        self.G = cfg.patch_grid  # (Gd, Gh, Gw)
        self.N = cfg.patch_grid[0] * cfg.patch_grid[1] * cfg.patch_grid[2]

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _sample_block_size(self, rng: np.random.Generator) -> Tuple[int, int, int]:
        """Sample a random 3D block size that covers the target fraction of N."""
        scale = rng.uniform(*self.cfg.target_scale_range)
        num_patches = max(1, int(round(scale * self.N)))

        # Sample log-uniform aspect ratios for the three axes
        log_lo = np.log(self.cfg.target_aspect_ratio[0])
        log_hi = np.log(self.cfg.target_aspect_ratio[1])
        ar_dh = np.exp(rng.uniform(log_lo, log_hi))
        ar_dw = np.exp(rng.uniform(log_lo, log_hi))

        # Solve for the cube root, then apply aspect ratios
        base = (num_patches / (ar_dh * ar_dw)) ** (1.0 / 3.0)
        bd = max(1, min(self.G[0], int(round(base))))
        bh = max(1, min(self.G[1], int(round(base * ar_dh))))
        bw = max(1, min(self.G[2], int(round(base * ar_dw))))

        return (bd, bh, bw)

    def _sample_block_mask(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a single target block and return a flat bool mask [N]."""
        bd, bh, bw = self._sample_block_size(rng)

        # Random top-left corner
        d0 = rng.integers(0, self.G[0] - bd + 1)
        h0 = rng.integers(0, self.G[1] - bh + 1)
        w0 = rng.integers(0, self.G[2] - bw + 1)

        mask_3d = np.zeros(self.G, dtype=bool)
        mask_3d[d0 : d0 + bd, h0 : h0 + bh, w0 : w0 + bw] = True
        return mask_3d.ravel()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def __call__(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        batch_size : int
        device : torch.device

        Returns
        -------
        context_mask : torch.BoolTensor  [B, N]
            True for patches visible to the context encoder.
        target_masks : torch.BoolTensor  [B, T, N]
            True for patches belonging to each target block.
        """
        rng = np.random.default_rng()
        T = self.cfg.num_target_blocks

        ctx_masks_np = np.empty((batch_size, self.N), dtype=bool)
        tgt_masks_np = np.empty((batch_size, T, self.N), dtype=bool)

        for b in range(batch_size):
            # --- Target blocks ---
            target_union = np.zeros(self.N, dtype=bool)
            for t in range(T):
                block = self._sample_block_mask(rng)
                tgt_masks_np[b, t] = block
                target_union |= block

            # --- Context mask = complement of target union ---
            ctx = ~target_union  # [N] bool — all non-target patches

            # Optionally sub-sample context to the scale range
            ctx_indices = np.flatnonzero(ctx)
            n_ctx_desired = int(
                rng.uniform(*self.cfg.context_scale_range) * self.N
            )
            # Don't exceed the available non-target patches
            n_ctx_keep = min(len(ctx_indices), n_ctx_desired)
            if n_ctx_keep < len(ctx_indices):
                drop = rng.choice(len(ctx_indices), len(ctx_indices) - n_ctx_keep, replace=False)
                ctx[ctx_indices[drop]] = False

            ctx_masks_np[b] = ctx

        context_mask = torch.from_numpy(ctx_masks_np).to(device)
        target_masks = torch.from_numpy(tgt_masks_np).to(device)
        return context_mask, target_masks
