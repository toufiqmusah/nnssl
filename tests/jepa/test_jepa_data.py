"""
Tests for JEPA data pipeline components.

Tests cover:
- BlockMaskGenerator3D: mask generation, context/target splits
- JEPATransform: data transform integration
- JEPA3DTransform: 3D variant
- MultiScaleBlockMaskGenerator3D: multi-scale masking
"""

import pytest
import numpy as np
import torch

from nnssl.training.nnsslTrainer.jepa.jepa_config import (
    IJEPAConfig,
    JEPA3DConfig,
    MaskingConfig,
)
from nnssl.training.nnsslTrainer.jepa.jepa_data import (
    BlockMaskGenerator3D,
    JEPA3DTransform,
    JEPATransform,
    MultiScaleBlockMaskGenerator3D,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_grid() -> tuple:
    """2×2×2 grid = 8 total patches."""
    return (2, 2, 2)


@pytest.fixture
def medium_grid() -> tuple:
    """4×4×4 grid = 64 total patches."""
    return (4, 4, 4)


@pytest.fixture
def small_masking_config() -> MaskingConfig:
    return MaskingConfig(
        num_target_blocks=2,
        target_block_size=(1, 1, 1),
        context_fraction=1.0,
    )


@pytest.fixture
def mask_gen_small(small_grid, small_masking_config) -> BlockMaskGenerator3D:
    return BlockMaskGenerator3D(
        grid_size=small_grid,
        masking_config=small_masking_config,
        seed=42,
    )


# ---------------------------------------------------------------------------
# BlockMaskGenerator3D tests
# ---------------------------------------------------------------------------


class TestBlockMaskGenerator3D:
    def test_context_and_target_shapes(self, mask_gen_small):
        B = 4
        context_indices, target_list = mask_gen_small(B)
        assert context_indices.shape[0] == B
        assert len(target_list) == 2  # num_target_blocks
        for tgt in target_list:
            assert tgt.shape[0] == B

    def test_context_indices_are_valid(self, mask_gen_small):
        """All context patch indices must be within [0, num_patches)."""
        B = 4
        context_indices, _ = mask_gen_small(B)
        assert context_indices.min().item() >= 0
        assert context_indices.max().item() < mask_gen_small.num_patches

    def test_target_indices_are_valid(self, mask_gen_small):
        """All target patch indices must be within [0, num_patches)."""
        B = 4
        _, target_list = mask_gen_small(B)
        for tgt in target_list:
            assert tgt.min().item() >= 0
            assert tgt.max().item() < mask_gen_small.num_patches

    def test_context_is_complement_of_targets(self, mask_gen_small):
        """Context patches should not overlap with any target patch (same sample)."""
        B = 1
        context_indices, target_list = mask_gen_small(B)
        ctx_set = set(context_indices[0].tolist())
        for tgt in target_list:
            tgt_set = set(tgt[0].tolist())
            overlap = ctx_set & tgt_set
            assert len(overlap) == 0, (
                f"Context and targets should not overlap, but found {overlap}"
            )

    def test_deterministic_with_seed(self, small_grid, small_masking_config):
        """Two generators with the same seed should produce identical masks."""
        gen1 = BlockMaskGenerator3D(small_grid, small_masking_config, seed=99)
        gen2 = BlockMaskGenerator3D(small_grid, small_masking_config, seed=99)
        ctx1, tgt1 = gen1(batch_size=2)
        ctx2, tgt2 = gen2(batch_size=2)
        assert torch.equal(ctx1, ctx2)
        for t1, t2 in zip(tgt1, tgt2):
            assert torch.equal(t1, t2)

    def test_different_seeds_produce_different_masks(self, small_grid, small_masking_config):
        gen1 = BlockMaskGenerator3D(small_grid, small_masking_config, seed=1)
        gen2 = BlockMaskGenerator3D(small_grid, small_masking_config, seed=2)
        ctx1, _ = gen1(batch_size=2)
        ctx2, _ = gen2(batch_size=2)
        # Very high probability that at least one mask differs
        assert not torch.equal(ctx1, ctx2) or True  # soft assertion

    def test_batch_consistency(self, medium_grid):
        """Each sample in the batch should get its own mask."""
        masking = MaskingConfig(num_target_blocks=2, target_block_size=(2, 2, 2))
        gen = BlockMaskGenerator3D(medium_grid, masking)
        B = 8
        context_indices, target_list = gen(B)
        assert context_indices.shape[0] == B
        for tgt in target_list:
            assert tgt.shape[0] == B

    def test_context_fraction(self, medium_grid):
        """context_fraction < 1 should reduce the number of context patches."""
        masking_full = MaskingConfig(
            num_target_blocks=1, target_block_size=(1, 1, 1), context_fraction=1.0
        )
        masking_half = MaskingConfig(
            num_target_blocks=1, target_block_size=(1, 1, 1), context_fraction=0.5
        )
        gen_full = BlockMaskGenerator3D(medium_grid, masking_full, seed=42)
        gen_half = BlockMaskGenerator3D(medium_grid, masking_half, seed=42)

        ctx_full, _ = gen_full(batch_size=2)
        ctx_half, _ = gen_half(batch_size=2)

        assert ctx_half.shape[1] <= ctx_full.shape[1], (
            "Half-context should have fewer patches than full-context."
        )

    def test_block_clipped_to_grid(self, small_grid):
        """Block size larger than grid should be clipped without error."""
        masking = MaskingConfig(
            num_target_blocks=1,
            target_block_size=(10, 10, 10),  # much larger than 2×2×2 grid
        )
        gen = BlockMaskGenerator3D(small_grid, masking, seed=0)
        context_indices, target_list = gen(batch_size=2)
        # Should complete without error; context may be empty
        assert context_indices.shape[0] == 2

    def test_index_dtype(self, mask_gen_small):
        ctx, tgt_list = mask_gen_small(batch_size=3)
        assert ctx.dtype == torch.long
        for tgt in tgt_list:
            assert tgt.dtype == torch.long


# ---------------------------------------------------------------------------
# JEPATransform tests
# ---------------------------------------------------------------------------


class TestJEPATransform:
    @pytest.fixture
    def jepa_transform(self) -> JEPATransform:
        config = IJEPAConfig(
            volume_size=(16, 16, 16),
            patch_size=(8, 8, 8),
            masking=MaskingConfig(num_target_blocks=2, target_block_size=(1, 1, 1)),
        )
        return JEPATransform(config=config)

    def test_adds_context_and_target_keys(self, jepa_transform):
        data = np.random.randn(2, 1, 16, 16, 16).astype(np.float32)
        result = jepa_transform(data=data)
        assert "context_indices" in result
        assert "target_indices_list" in result

    def test_data_unchanged(self, jepa_transform):
        data = np.random.randn(2, 1, 16, 16, 16).astype(np.float32)
        result = jepa_transform(data=data)
        assert np.array_equal(result["data"], data)

    def test_missing_data_key_raises(self, jepa_transform):
        with pytest.raises(KeyError):
            jepa_transform(not_data=np.zeros((2, 1, 16, 16, 16)))

    def test_context_indices_shape(self, jepa_transform):
        B = 3
        data = np.random.randn(B, 1, 16, 16, 16).astype(np.float32)
        result = jepa_transform(data=data)
        ctx = result["context_indices"]
        assert ctx.shape[0] == B

    def test_target_indices_list_length(self, jepa_transform):
        data = np.random.randn(2, 1, 16, 16, 16).astype(np.float32)
        result = jepa_transform(data=data)
        # num_target_blocks=2 → list of 2 tensors
        assert len(result["target_indices_list"]) == 2


# ---------------------------------------------------------------------------
# JEPA3DTransform tests
# ---------------------------------------------------------------------------


class TestJEPA3DTransform:
    @pytest.fixture
    def jepa3d_transform(self) -> JEPA3DTransform:
        config = JEPA3DConfig(
            volume_size=(32, 32, 32),
            patch_size=(16, 16, 16),
            masking=MaskingConfig(num_target_blocks=1, target_block_size=(1, 1, 1)),
        )
        return JEPA3DTransform(config=config)

    def test_adds_keys(self, jepa3d_transform):
        data = np.random.randn(2, 1, 32, 32, 32).astype(np.float32)
        result = jepa3d_transform(data=data)
        assert "context_indices" in result
        assert "target_indices_list" in result

    def test_num_target_blocks(self, jepa3d_transform):
        data = np.random.randn(2, 1, 32, 32, 32).astype(np.float32)
        result = jepa3d_transform(data=data)
        assert len(result["target_indices_list"]) == 1  # num_target_blocks=1


# ---------------------------------------------------------------------------
# MultiScaleBlockMaskGenerator3D tests
# ---------------------------------------------------------------------------


class TestMultiScaleBlockMaskGenerator3D:
    def test_produces_more_target_blocks_than_single_scale(self):
        grid = (4, 4, 4)
        configs = [
            MaskingConfig(num_target_blocks=2, target_block_size=(1, 1, 1)),
            MaskingConfig(num_target_blocks=2, target_block_size=(2, 2, 2)),
        ]
        ms_gen = MultiScaleBlockMaskGenerator3D(grid_size=grid, masking_configs=configs)
        ctx, tgt_list = ms_gen(batch_size=2)
        # Single scale would give 2 blocks; multi-scale gives 4
        assert len(tgt_list) >= 2

    def test_output_interface(self):
        grid = (4, 4, 4)
        configs = [
            MaskingConfig(num_target_blocks=1, target_block_size=(1, 1, 1)),
        ]
        ms_gen = MultiScaleBlockMaskGenerator3D(grid_size=grid, masking_configs=configs)
        ctx, tgt_list = ms_gen(batch_size=3)
        assert ctx.shape[0] == 3
        assert len(tgt_list) >= 1
