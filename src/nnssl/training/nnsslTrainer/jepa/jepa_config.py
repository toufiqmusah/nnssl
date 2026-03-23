"""
Configuration dataclasses for JEPA (Joint-Embedding Predictive Architecture) methods.

Supports both I-JEPA (patch-based, Vision Transformer) and 3D-JEPA (volumetric, CNN-based)
for 3D medical image self-supervised pretraining.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class VICRegLossConfig:
    """Configuration for VICReg (Variance-Invariance-Covariance Regularization) loss.

    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICLR 2022.
    """

    variance_weight: float = 25.0
    """Weight for the variance term (lambda). Encourages non-collapsed representations."""

    invariance_weight: float = 25.0
    """Weight for the invariance term (mu). Makes representations predictive."""

    covariance_weight: float = 1.0
    """Weight for the covariance term (nu). Decorrelates representation dimensions."""

    variance_threshold: float = 1.0
    """Target standard deviation per dimension (gamma). Prevents variance collapse."""

    epsilon: float = 1e-4
    """Small constant for numerical stability in variance computation."""


@dataclass
class PredictorConfig:
    """Configuration for the JEPA predictor network.

    The predictor is intentionally narrow (fewer parameters than the encoder) to
    encourage non-trivial representations and prevent shortcuts.
    """

    hidden_dim: int = 384
    """Hidden dimension of the predictor Transformer."""

    num_blocks: int = 6
    """Number of Transformer blocks in the predictor."""

    num_heads: int = 6
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """Ratio of MLP hidden dim to embed dim."""

    dropout: float = 0.0
    """Dropout rate."""


@dataclass
class EncoderConfig:
    """Configuration for the JEPA Vision Transformer encoder (context and target)."""

    embed_dim: int = 384
    """Token embedding dimension (ViT-S scale by default)."""

    depth: int = 6
    """Number of Transformer blocks."""

    num_heads: int = 6
    """Number of attention heads."""

    mlp_ratio: float = 4.0
    """Ratio of MLP hidden dim to embed dim."""

    dropout: float = 0.0
    """Dropout rate."""


@dataclass
class MaskingConfig:
    """Configuration for the 3D block masking strategy used in I-JEPA.

    Masking operates at the patch level (not individual voxels) to create
    semantically meaningful prediction targets.
    """

    num_target_blocks: int = 4
    """Number of target blocks to predict per training step."""

    target_block_size: Tuple[int, int, int] = (2, 2, 2)
    """Size of each target block in patch units (dx, dy, dz)."""

    context_fraction: float = 1.0
    """Fraction of non-target patches used as context (1.0 = use all)."""


@dataclass
class IJEPAConfig:
    """Complete configuration for I-JEPA (Image JEPA) training.

    I-JEPA learns representations by predicting latent representations of
    target image regions from context regions using a Vision Transformer.

    Reference: Assran et al., "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture", CVPR 2023.
    """

    # ------- Volume and patch dimensions ------- #
    volume_size: Tuple[int, int, int] = (96, 96, 96)
    """Input 3D volume dimensions (X, Y, Z) in voxels."""

    patch_size: Tuple[int, int, int] = (16, 16, 16)
    """Size of each non-overlapping 3D patch (Px, Py, Pz) in voxels."""

    in_channels: int = 1
    """Number of input channels (1 for grayscale medical images)."""

    # ------- Encoder configuration ------- #
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    """Configuration for both context and target encoder."""

    # ------- Predictor configuration ------- #
    predictor: PredictorConfig = field(default_factory=PredictorConfig)
    """Configuration for the predictor network."""

    # ------- Masking configuration ------- #
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    """Configuration for the block masking strategy."""

    # ------- Loss configuration ------- #
    loss: VICRegLossConfig = field(default_factory=VICRegLossConfig)
    """Configuration for the VICReg loss function."""

    # ------- Momentum encoder schedule ------- #
    momentum_start: float = 0.996
    """Initial momentum for exponential moving average (EMA) target encoder updates."""

    momentum_end: float = 1.0
    """Final momentum value at end of training (cosine schedule)."""

    use_momentum_schedule: bool = True
    """Whether to apply a cosine schedule to the EMA momentum coefficient."""


@dataclass
class JEPA3DConfig(IJEPAConfig):
    """Configuration for 3D-JEPA with volumetric CNN-based processing.

    3D-JEPA extends I-JEPA to exploit full 3D spatial coherence in medical
    images using 3D convolutional encoders and multi-scale patch hierarchies.
    """

    volume_size: Tuple[int, int, int] = (128, 128, 128)
    """Input 3D volume dimensions. Larger than I-JEPA for volumetric context."""

    patch_size: Tuple[int, int, int] = (16, 16, 16)
    """3D patch size. Yields 8x8x8 = 512 patches for a 128^3 volume."""

    encoder: EncoderConfig = field(
        default_factory=lambda: EncoderConfig(
            embed_dim=512,
            depth=8,
            num_heads=8,
        )
    )
    """Larger encoder for full volumetric 3D processing."""

    masking: MaskingConfig = field(
        default_factory=lambda: MaskingConfig(
            num_target_blocks=4,
            target_block_size=(3, 3, 3),
        )
    )
    """Larger target blocks for volumetric coherence."""

    momentum_start: float = 0.998
    """Higher initial momentum for volumetric 3D processing."""
