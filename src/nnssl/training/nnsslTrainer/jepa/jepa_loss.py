"""
VICReg (Variance-Invariance-Covariance Regularization) loss for JEPA.

VICReg prevents representational collapse by simultaneously:
  - Maintaining sufficient variance across the batch (variance term)
  - Making predictions match targets (invariance term)
  - Decorrelating the embedding dimensions (covariance term)

Reference:
    Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICLR 2022.
    https://arxiv.org/abs/2105.04906
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnssl.training.nnsslTrainer.jepa.jepa_config import VICRegLossConfig


class VICRegLoss(nn.Module):
    """VICReg loss function.

    Computes the three VICReg regularization terms and combines them:

        L = λ * L_var + μ * L_inv + ν * L_cov

    where:
        L_var: Variance term — penalises low per-dimension std across the batch.
        L_inv: Invariance term — MSE between two paired representations
               (prediction and target in the JEPA setting).
        L_cov: Covariance term — penalises non-zero off-diagonal elements of
               the empirical covariance matrix.

    Args:
        config: VICRegLossConfig dataclass.
    """

    def __init__(self, config: VICRegLossConfig) -> None:
        super().__init__()
        self.variance_weight = config.variance_weight
        self.invariance_weight = config.invariance_weight
        self.covariance_weight = config.covariance_weight
        self.variance_threshold = config.variance_threshold
        self.epsilon = config.epsilon

    # ------------------------------------------------------------------
    # Individual loss terms
    # ------------------------------------------------------------------

    def variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Variance regularisation term.

        Encourages each embedding dimension to maintain a standard deviation
        above ``variance_threshold`` across the batch, preventing collapse to
        constant representations.

        .. math::
            L_{\\text{var}}(Z) = \\frac{1}{D} \\sum_{j=1}^{D}
                \\max(0,\\; \\gamma - \\text{Std}(Z_{:,j}))

        Args:
            z: Batch of representations, shape ``(N, D)``.

        Returns:
            Scalar variance loss.
        """
        # z: (N, D)
        std = torch.sqrt(z.var(dim=0) + self.epsilon)  # (D,)
        loss = F.relu(self.variance_threshold - std).mean()
        return loss

    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Invariance (prediction) term.

        Mean squared error between two sets of representations.  In the JEPA
        context ``z1`` is the predictor output and ``z2`` is the (detached)
        target encoder output.

        .. math::
            L_{\\text{inv}}(Z_1, Z_2) = \\frac{1}{N} \\|Z_1 - Z_2\\|_F^2

        Args:
            z1: Predicted representations, shape ``(N, D)``.
            z2: Target representations, shape ``(N, D)``.

        Returns:
            Scalar invariance loss.
        """
        return F.mse_loss(z1, z2)

    def covariance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """Covariance regularisation term.

        Penalises non-zero off-diagonal elements of the empirical covariance
        matrix, decorrelating the embedding dimensions and discouraging
        representational redundancy.

        .. math::
            L_{\\text{cov}}(Z) = \\frac{1}{D}
                \\sum_{i \\neq j} \\left[\\text{Cov}(Z)\\right]_{i,j}^2

        Args:
            z: Batch of representations, shape ``(N, D)``.

        Returns:
            Scalar covariance loss.
        """
        N, D = z.shape
        z_centered = z - z.mean(dim=0, keepdim=True)  # (N, D)
        cov = (z_centered.T @ z_centered) / (N - 1)  # (D, D)

        # Sum of squared off-diagonal elements
        diag_mask = torch.eye(D, device=z.device, dtype=torch.bool)
        off_diag_sq_sum = cov[~diag_mask].pow(2).sum()
        return off_diag_sq_sum / D

    # ------------------------------------------------------------------
    # Aggregated forward
    # ------------------------------------------------------------------

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the full VICReg loss.

        Applies variance and covariance terms to *both* representations and
        the invariance (MSE) term between them.

        Args:
            z1: First set of representations, shape ``(N, D)``.
                In JEPA: predictor output flattened over the batch dimension.
            z2: Second set of representations, shape ``(N, D)``.
                In JEPA: target encoder output (detached).

        Returns:
            Tuple of four scalars:
                - ``total_loss``: Weighted sum of all three terms.
                - ``loss_var``: Unweighted variance loss (for logging).
                - ``loss_inv``: Unweighted invariance loss (for logging).
                - ``loss_cov``: Unweighted covariance loss (for logging).
        """
        # Reshape to (N, D) if needed (e.g. flattened batch × tokens)
        assert z1.shape == z2.shape, (
            f"z1 and z2 must have the same shape, got {z1.shape} vs {z2.shape}"
        )
        if z1.dim() > 2:
            z1 = z1.reshape(-1, z1.shape[-1])
            z2 = z2.reshape(-1, z2.shape[-1])

        loss_var = self.variance_loss(z1) + self.variance_loss(z2)
        loss_inv = self.invariance_loss(z1, z2)
        loss_cov = self.covariance_loss(z1) + self.covariance_loss(z2)

        total_loss = (
            self.variance_weight * loss_var
            + self.invariance_weight * loss_inv
            + self.covariance_weight * loss_cov
        )
        return total_loss, loss_var, loss_inv, loss_cov
