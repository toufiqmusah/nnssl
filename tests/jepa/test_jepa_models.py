"""
Tests for JEPA core model components.

Tests cover:
- PatchEmbed3D: patch extraction and embedding
- build_3d_sincos_pos_embed: positional embedding generation
- VisionTransformer3D: full forward and masked forward passes
- JEPAPredictor: prediction from context tokens
- IJEPAModel: complete forward pass and EMA update
- JEPA3DModel: CNN-based model forward pass and EMA update
"""

import pytest
import torch

from nnssl.training.nnsslTrainer.jepa.jepa_config import (
    EncoderConfig,
    IJEPAConfig,
    JEPA3DConfig,
    MaskingConfig,
    PredictorConfig,
    VICRegLossConfig,
)
from nnssl.training.nnsslTrainer.jepa.jepa_models import (
    JEPA3DModel,
    IJEPAModel,
    JEPAPredictor,
    PatchEmbed3D,
    VisionTransformer3D,
    build_3d_sincos_pos_embed,
)


# ---------------------------------------------------------------------------
# Fixtures: small configs to keep tests fast on CPU
# ---------------------------------------------------------------------------


@pytest.fixture
def small_encoder_config() -> EncoderConfig:
    return EncoderConfig(embed_dim=48, depth=2, num_heads=4, mlp_ratio=2.0)


@pytest.fixture
def small_predictor_config() -> PredictorConfig:
    return PredictorConfig(hidden_dim=48, num_blocks=2, num_heads=4, mlp_ratio=2.0)


@pytest.fixture
def small_ijepa_config(small_encoder_config, small_predictor_config) -> IJEPAConfig:
    """Small I-JEPA config with 4³ volume, 2³ patches → 2×2×2 = 8 total patches."""
    return IJEPAConfig(
        volume_size=(16, 16, 16),
        patch_size=(8, 8, 8),
        in_channels=1,
        encoder=small_encoder_config,
        predictor=small_predictor_config,
        masking=MaskingConfig(num_target_blocks=2, target_block_size=(1, 1, 1)),
        loss=VICRegLossConfig(),
        momentum_start=0.996,
        momentum_end=1.0,
        use_momentum_schedule=False,
    )


@pytest.fixture
def small_jepa3d_config(small_encoder_config, small_predictor_config) -> JEPA3DConfig:
    """Small 3D-JEPA config with CNN encoder."""
    return JEPA3DConfig(
        volume_size=(32, 32, 32),
        patch_size=(16, 16, 16),
        in_channels=1,
        encoder=EncoderConfig(embed_dim=48, depth=2, num_heads=4),
        predictor=small_predictor_config,
        masking=MaskingConfig(num_target_blocks=1, target_block_size=(1, 1, 1)),
        loss=VICRegLossConfig(),
        momentum_start=0.996,
    )


# ---------------------------------------------------------------------------
# PatchEmbed3D tests
# ---------------------------------------------------------------------------


class TestPatchEmbed3D:
    def test_output_shape(self):
        embed = PatchEmbed3D(
            volume_size=(32, 32, 32),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=64,
        )
        x = torch.randn(2, 1, 32, 32, 32)
        out = embed(x)
        # 32/8 = 4 patches per axis → 4³ = 64 total patches
        assert out.shape == (2, 64, 64), f"Expected (2, 64, 64), got {out.shape}"

    def test_patch_count_matches_grid(self):
        embed = PatchEmbed3D(
            volume_size=(16, 16, 16),
            patch_size=(8, 8, 8),
            in_channels=1,
            embed_dim=32,
        )
        assert embed.num_patches == 8
        assert embed.grid_size == (2, 2, 2)

    def test_invalid_volume_size_raises(self):
        with pytest.raises(AssertionError):
            PatchEmbed3D(
                volume_size=(15, 16, 16),  # 15 not divisible by 8
                patch_size=(8, 8, 8),
                in_channels=1,
                embed_dim=32,
            )


# ---------------------------------------------------------------------------
# Positional embedding tests
# ---------------------------------------------------------------------------


class TestSinCos3DPosEmbed:
    def test_output_shape(self):
        embed = build_3d_sincos_pos_embed(embed_dim=48, grid_size=(2, 2, 2))
        assert embed.shape == (1, 8, 48)

    def test_embed_dim_divisibility(self):
        with pytest.raises(AssertionError):
            build_3d_sincos_pos_embed(embed_dim=0, grid_size=(2, 2, 2))  # 0 gives sincos_dim=0

    def test_non_divisible_embed_dim_pads(self):
        """Non-multiples of 6 should be padded to the target embed_dim."""
        embed = build_3d_sincos_pos_embed(embed_dim=10, grid_size=(2, 2, 2))
        assert embed.shape == (1, 8, 10)

    def test_values_bounded(self):
        embed = build_3d_sincos_pos_embed(embed_dim=48, grid_size=(3, 3, 3))
        # Sinusoidal values are in [-1, 1]
        assert embed.abs().max().item() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# VisionTransformer3D tests
# ---------------------------------------------------------------------------


class TestVisionTransformer3D:
    def test_forward_full_shape(self, small_encoder_config):
        vit = VisionTransformer3D(
            volume_size=(16, 16, 16),
            patch_size=(8, 8, 8),
            in_channels=1,
            config=small_encoder_config,
        )
        x = torch.randn(2, 1, 16, 16, 16)
        out = vit.forward_full(x)
        # 2 patches per axis → 8 patches total
        assert out.shape == (2, 8, 48), f"Expected (2, 8, 48), got {out.shape}"

    def test_forward_masked_shape(self, small_encoder_config):
        vit = VisionTransformer3D(
            volume_size=(16, 16, 16),
            patch_size=(8, 8, 8),
            in_channels=1,
            config=small_encoder_config,
        )
        x = torch.randn(2, 1, 16, 16, 16)
        # Keep 5 out of 8 patches
        keep_indices = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]], dtype=torch.long)
        ctx_tokens, ctx_pos = vit.forward_masked(x, keep_indices)
        assert ctx_tokens.shape == (2, 5, 48)
        assert ctx_pos.shape == (2, 5, 48)

    def test_no_grad_in_target_encoder(self, small_encoder_config):
        """Verify that the target encoder has no gradients (requires_grad=False)."""
        from nnssl.training.nnsslTrainer.jepa.jepa_models import IJEPAModel

        cfg = IJEPAConfig(
            volume_size=(16, 16, 16),
            patch_size=(8, 8, 8),
            in_channels=1,
            encoder=small_encoder_config,
            predictor=PredictorConfig(hidden_dim=48, num_blocks=2, num_heads=4),
        )
        model = IJEPAModel(config=cfg)
        for param in model.target_encoder.parameters():
            assert not param.requires_grad, "Target encoder must not require gradients."


# ---------------------------------------------------------------------------
# JEPAPredictor tests
# ---------------------------------------------------------------------------


class TestJEPAPredictor:
    def test_forward_shape(self, small_predictor_config):
        predictor = JEPAPredictor(encoder_dim=48, config=small_predictor_config)
        B, n_ctx, n_tgt, D = 2, 5, 3, 48
        ctx_tokens = torch.randn(B, n_ctx, D)
        ctx_pos = torch.randn(B, n_ctx, D)
        tgt_pos = torch.randn(B, n_tgt, D)
        out = predictor(ctx_tokens, ctx_pos, tgt_pos)
        assert out.shape == (B, n_tgt, D), f"Expected ({B}, {n_tgt}, {D}), got {out.shape}"

    def test_output_is_finite(self, small_predictor_config):
        predictor = JEPAPredictor(encoder_dim=48, config=small_predictor_config)
        ctx_tokens = torch.randn(2, 5, 48)
        out = predictor(ctx_tokens, ctx_tokens, ctx_tokens[:, :3, :])
        assert torch.isfinite(out).all(), "Predictor output contains non-finite values."


# ---------------------------------------------------------------------------
# IJEPAModel tests
# ---------------------------------------------------------------------------


class TestIJEPAModel:
    def test_forward_output_shapes(self, small_ijepa_config):
        model = IJEPAModel(config=small_ijepa_config)
        B = 2
        x = torch.randn(B, 1, 16, 16, 16)

        # 8 patches total: keep 6 as context, 2 target blocks of 1 patch each
        context_indices = torch.tensor([[0, 1, 2, 3, 4, 5]] * B, dtype=torch.long)
        target_indices_list = [
            torch.tensor([[6]] * B, dtype=torch.long),
            torch.tensor([[7]] * B, dtype=torch.long),
        ]
        preds, targets = model(x, context_indices, target_indices_list)

        assert len(preds) == 2
        assert len(targets) == 2
        for pred, tgt in zip(preds, targets):
            assert pred.shape == (B, 1, 48)
            assert tgt.shape == (B, 1, 48)

    def test_targets_have_no_grad(self, small_ijepa_config):
        model = IJEPAModel(config=small_ijepa_config)
        x = torch.randn(2, 1, 16, 16, 16)
        context_indices = torch.tensor([[0, 1, 2, 3, 4, 5]] * 2, dtype=torch.long)
        target_indices_list = [torch.tensor([[6]] * 2, dtype=torch.long)]
        _, targets = model(x, context_indices, target_indices_list)
        for tgt in targets:
            assert not tgt.requires_grad, "Targets must be detached from the computational graph."

    def test_ema_update_changes_target_encoder(self, small_ijepa_config):
        """EMA update should change target encoder weights."""
        model = IJEPAModel(config=small_ijepa_config)

        # Capture initial target encoder weights
        before = [p.data.clone() for p in model.target_encoder.parameters()]

        # Perturb context encoder
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.data += 0.1

        model.update_target_encoder(momentum=0.9)
        after = [p.data.clone() for p in model.target_encoder.parameters()]

        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed, "Target encoder should change after EMA update."

    def test_ema_update_with_momentum_one_no_change(self, small_ijepa_config):
        """With momentum=1.0, target encoder should not change."""
        model = IJEPAModel(config=small_ijepa_config)
        before = [p.data.clone() for p in model.target_encoder.parameters()]
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.data += 1.0
        model.update_target_encoder(momentum=1.0)
        for b, p in zip(before, model.target_encoder.parameters()):
            assert torch.allclose(b, p.data), "Momentum=1.0 should leave target encoder unchanged."

    def test_num_patches_property(self, small_ijepa_config):
        model = IJEPAModel(config=small_ijepa_config)
        assert model.num_patches == 8  # (16/8)^3 = 2^3 = 8


# ---------------------------------------------------------------------------
# JEPA3DModel tests
# ---------------------------------------------------------------------------


class TestJEPA3DModel:
    def test_forward_output_shapes(self, small_jepa3d_config):
        model = JEPA3DModel(config=small_jepa3d_config)
        B = 2
        x = torch.randn(B, 1, 32, 32, 32)

        # Grid: 32/16 = 2 patches per axis → 8 total
        context_indices = torch.tensor([[0, 1, 2, 3, 4, 5, 6]] * B, dtype=torch.long)
        target_indices_list = [torch.tensor([[7]] * B, dtype=torch.long)]

        preds, targets = model(x, context_indices, target_indices_list)
        assert len(preds) == 1
        D = small_jepa3d_config.encoder.embed_dim
        assert preds[0].shape == (B, 1, D)
        assert targets[0].shape == (B, 1, D)

    def test_ema_update(self, small_jepa3d_config):
        model = JEPA3DModel(config=small_jepa3d_config)
        before = [p.data.clone() for p in model.target_encoder.parameters()]
        with torch.no_grad():
            for p in model.context_encoder.parameters():
                p.data += 0.1
        model.update_target_encoder(momentum=0.9)
        after = [p.data.clone() for p in model.target_encoder.parameters()]
        changed = any(not torch.allclose(b, a) for b, a in zip(before, after))
        assert changed, "Target encoder should change after EMA update."

    def test_target_encoder_no_grad(self, small_jepa3d_config):
        model = JEPA3DModel(config=small_jepa3d_config)
        for param in model.target_encoder.parameters():
            assert not param.requires_grad
