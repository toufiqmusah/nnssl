"""
Tests for VICReg loss implementation.

Tests cover:
- Individual loss terms (variance, invariance, covariance)
- Combined forward pass
- Collapse prevention properties
- Loss term weighting
- Numerical stability
"""

import pytest
import torch
import torch.nn as nn

from nnssl.training.nnsslTrainer.jepa.jepa_config import VICRegLossConfig
from nnssl.training.nnsslTrainer.jepa.jepa_loss import VICRegLoss


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> VICRegLossConfig:
    return VICRegLossConfig(
        variance_weight=25.0,
        invariance_weight=25.0,
        covariance_weight=1.0,
        variance_threshold=1.0,
        epsilon=1e-4,
    )


@pytest.fixture
def loss_fn(default_config) -> VICRegLoss:
    return VICRegLoss(config=default_config)


# ---------------------------------------------------------------------------
# Variance loss tests
# ---------------------------------------------------------------------------


class TestVarianceLoss:
    def test_zero_when_std_above_threshold(self, loss_fn):
        """Variance loss is 0 when std per dimension is >= threshold."""
        # Create z with std=2 per dimension
        N, D = 32, 16
        z = torch.randn(N, D) * 2.0  # std ≈ 2 > threshold (1.0)
        loss = loss_fn.variance_loss(z)
        assert loss.item() == pytest.approx(0.0, abs=0.2), (
            f"Expected ~0 variance loss, got {loss.item()}"
        )

    def test_positive_when_std_below_threshold(self, loss_fn):
        """Variance loss is positive when std < threshold."""
        N, D = 32, 16
        z = torch.full((N, D), 0.5)  # constant → std=0 < threshold (1.0)
        loss = loss_fn.variance_loss(z)
        assert loss.item() > 0.0

    def test_collapsed_representation(self, loss_fn):
        """Completely collapsed representation (std=0) gives maximum variance loss."""
        N, D = 32, 16
        z = torch.ones(N, D)
        loss = loss_fn.variance_loss(z)
        # With epsilon=1e-4: std = sqrt(0 + 1e-4) ≈ 0.01, so loss ≈ threshold - 0.01 ≈ 0.99
        assert loss.item() == pytest.approx(1.0, abs=0.02), (
            f"Expected variance loss ≈ 1.0 for collapsed repr, got {loss.item()}"
        )


# ---------------------------------------------------------------------------
# Invariance loss tests
# ---------------------------------------------------------------------------


class TestInvarianceLoss:
    def test_zero_for_identical_inputs(self, loss_fn):
        z = torch.randn(16, 32)
        loss = loss_fn.invariance_loss(z, z)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_positive_for_different_inputs(self, loss_fn):
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        loss = loss_fn.invariance_loss(z1, z2)
        assert loss.item() > 0.0

    def test_mse_correctness(self, loss_fn):
        """Invariance loss equals PyTorch MSE loss."""
        z1 = torch.randn(16, 32)
        z2 = torch.randn(16, 32)
        expected = nn.functional.mse_loss(z1, z2)
        assert loss_fn.invariance_loss(z1, z2).item() == pytest.approx(expected.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Covariance loss tests
# ---------------------------------------------------------------------------


class TestCovarianceLoss:
    def test_zero_for_diagonal_covariance(self, loss_fn):
        """Covariance loss is near 0 if dimensions are uncorrelated."""
        N, D = 256, 16
        # Independent standard normal columns → cov ≈ identity
        z = torch.randn(N, D)
        loss = loss_fn.covariance_loss(z)
        # Off-diagonal should be small but not exactly 0
        assert loss.item() < 1.0, f"Expected small cov loss, got {loss.item()}"

    def test_positive_for_correlated_dimensions(self, loss_fn):
        """Covariance loss is positive when dimensions are correlated."""
        N, D = 64, 4
        # All dimensions are identical → maximally correlated
        base = torch.randn(N, 1).expand(N, D)
        loss = loss_fn.covariance_loss(base)
        assert loss.item() > 0.0

    def test_output_is_scalar(self, loss_fn):
        z = torch.randn(32, 16)
        loss = loss_fn.covariance_loss(z)
        assert loss.dim() == 0


# ---------------------------------------------------------------------------
# Combined forward pass tests
# ---------------------------------------------------------------------------


class TestVICRegLossForward:
    def test_returns_four_values(self, loss_fn):
        z1 = torch.randn(32, 16)
        z2 = torch.randn(32, 16)
        result = loss_fn(z1, z2)
        assert len(result) == 4

    def test_total_loss_is_weighted_sum(self, default_config):
        """Total loss = λ*var + μ*inv + ν*cov."""
        loss_fn = VICRegLoss(config=default_config)
        z1 = torch.randn(32, 16)
        z2 = torch.randn(32, 16)
        total, l_var, l_inv, l_cov = loss_fn(z1, z2)
        expected = (
            default_config.variance_weight * l_var
            + default_config.invariance_weight * l_inv
            + default_config.covariance_weight * l_cov
        )
        assert total.item() == pytest.approx(expected.item(), rel=1e-5)

    def test_all_losses_non_negative(self, loss_fn):
        z1 = torch.randn(32, 16)
        z2 = torch.randn(32, 16)
        total, l_var, l_inv, l_cov = loss_fn(z1, z2)
        assert total.item() >= 0.0
        assert l_var.item() >= 0.0
        assert l_inv.item() >= 0.0
        assert l_cov.item() >= 0.0

    def test_identical_inputs_zero_invariance_loss(self, loss_fn):
        z = torch.randn(32, 16) * 2.0  # high variance to suppress var term
        total, l_var, l_inv, l_cov = loss_fn(z, z)
        assert l_inv.item() == pytest.approx(0.0, abs=1e-5)

    def test_3d_inputs_flattened(self, loss_fn):
        """Loss should handle 3D inputs (B, N, D) by flattening to (B*N, D)."""
        z1 = torch.randn(4, 8, 16)
        z2 = torch.randn(4, 8, 16)
        total, l_var, l_inv, l_cov = loss_fn(z1, z2)
        assert total.item() >= 0.0

    def test_shape_mismatch_raises(self, loss_fn):
        z1 = torch.randn(32, 16)
        z2 = torch.randn(32, 8)  # wrong D
        with pytest.raises(AssertionError):
            loss_fn(z1, z2)

    def test_gradients_flow_through_loss(self, loss_fn):
        """Verify that gradients can be computed through the loss."""
        z1 = torch.randn(16, 8, requires_grad=True)
        z2 = torch.randn(16, 8)
        total, *_ = loss_fn(z1, z2)
        total.backward()
        assert z1.grad is not None
        assert torch.isfinite(z1.grad).all()

    def test_numerical_stability(self, loss_fn):
        """Loss should be finite even for very small values."""
        z1 = torch.zeros(16, 8)
        z2 = torch.zeros(16, 8)
        total, l_var, l_inv, l_cov = loss_fn(z1, z2)
        assert torch.isfinite(total)

    def test_collapse_prevention_property(self):
        """With high variance weight, loss should be lower when representations don't collapse."""
        config = VICRegLossConfig(variance_weight=100.0, invariance_weight=0.0, covariance_weight=0.0)
        loss_fn = VICRegLoss(config=config)

        # Non-collapsed: high std
        z_good = torch.randn(64, 32) * 2.0
        total_good, _, _, _ = loss_fn(z_good, z_good)

        # Collapsed: std near 0
        z_bad = torch.ones(64, 32) * 0.001
        total_bad, _, _, _ = loss_fn(z_bad, z_bad)

        assert total_bad.item() > total_good.item(), (
            "Collapsed representations should have higher loss than non-collapsed."
        )
