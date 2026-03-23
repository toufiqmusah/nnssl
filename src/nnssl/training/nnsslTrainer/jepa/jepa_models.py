"""
Core JEPA model architectures for 3D medical image self-supervised pretraining.

Implements:
- PatchEmbed3D: Non-overlapping 3D patch embedding (patchification)
- VisionTransformer3D: 3D Vision Transformer backbone
- ContextEncoder3D: Processes visible (context) patches only
- TargetEncoder3D: EMA-updated encoder, processes all patches
- JEPAPredictor: Predicts target encoder representations from context
- IJEPAModel: Complete I-JEPA model
- JEPA3DModel: 3D volumetric JEPA model

References:
    Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture", CVPR 2023.
"""

import math
from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from nnssl.training.nnsslTrainer.jepa.jepa_config import (
    EncoderConfig,
    IJEPAConfig,
    JEPA3DConfig,
    PredictorConfig,
)


# ---------------------------------------------------------------------------
# 3D Positional Embedding
# ---------------------------------------------------------------------------


def build_3d_sincos_pos_embed(
    embed_dim: int,
    grid_size: Tuple[int, int, int],
    temperature: float = 10000.0,
) -> torch.Tensor:
    """Generate 3D sinusoidal positional embeddings.

    Splits the embedding dimension evenly across the three spatial axes
    (X, Y, Z), so the model can distinguish patch positions in 3D space.
    If ``embed_dim`` is not divisible by 6, the embedding is padded with
    zeros to reach the target dimension.

    Args:
        embed_dim: Total embedding dimension.
        grid_size: Number of patches along each axis (gX, gY, gZ).
        temperature: Temperature for the sinusoidal functions.

    Returns:
        pos_embed: Positional embeddings of shape (1, gX*gY*gZ, embed_dim).
    """
    # Largest multiple of 6 that is <= embed_dim (used for sincos part)
    sincos_dim = (embed_dim // 6) * 6
    assert sincos_dim > 0, (
        f"embed_dim ({embed_dim}) must be at least 6 for 3D sincos pos embed."
    )
    gx, gy, gz = grid_size
    dim_per_axis = sincos_dim // 6  # 2 * dim_per_axis per spatial axis

    # Build coordinate grids
    gx_pos = torch.arange(gx, dtype=torch.float32)
    gy_pos = torch.arange(gy, dtype=torch.float32)
    gz_pos = torch.arange(gz, dtype=torch.float32)

    # Omega for sinusoidal encoding
    dim_idx = torch.arange(dim_per_axis, dtype=torch.float32)
    omega = 1.0 / (temperature ** (dim_idx / dim_per_axis))  # (dim_per_axis,)

    # Compute sin/cos for each axis: shapes (g_i, dim_per_axis)
    sincos_x = torch.cat(
        [torch.sin(torch.outer(gx_pos, omega)), torch.cos(torch.outer(gx_pos, omega))], dim=1
    )  # (gX, 2*dim_per_axis)
    sincos_y = torch.cat(
        [torch.sin(torch.outer(gy_pos, omega)), torch.cos(torch.outer(gy_pos, omega))], dim=1
    )  # (gY, 2*dim_per_axis)
    sincos_z = torch.cat(
        [torch.sin(torch.outer(gz_pos, omega)), torch.cos(torch.outer(gz_pos, omega))], dim=1
    )  # (gZ, 2*dim_per_axis)

    # Expand and combine over the 3D grid via broadcasting
    # Each shape: (gX, gY, gZ, 2*dim_per_axis) before reshaping
    sx = sincos_x[:, None, None, :].expand(gx, gy, gz, -1)  # (gX, gY, gZ, d)
    sy = sincos_y[None, :, None, :].expand(gx, gy, gz, -1)  # (gX, gY, gZ, d)
    sz = sincos_z[None, None, :, :].expand(gx, gy, gz, -1)  # (gX, gY, gZ, d)

    # Concatenate along last dim → (gX, gY, gZ, sincos_dim)
    pos_embed = torch.cat([sx, sy, sz], dim=-1)
    pos_embed = pos_embed.reshape(1, gx * gy * gz, sincos_dim)  # (1, N, sincos_dim)

    # Pad to embed_dim if necessary
    if sincos_dim < embed_dim:
        padding = torch.zeros(1, gx * gy * gz, embed_dim - sincos_dim)
        pos_embed = torch.cat([pos_embed, padding], dim=-1)  # (1, N, embed_dim)

    return pos_embed


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------


class Attention(nn.Module):
    """Multi-head self-attention module."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim ({dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, N, D).

        Returns:
            Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, N, H, head_dim)
        q = q.transpose(1, 2)  # (B, H, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """Feed-forward network used inside each Transformer block."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Standard Transformer block: LayerNorm → Attention → residual → LayerNorm → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, hidden_features=int(dim * mlp_ratio), dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 3D Patch Embedding
# ---------------------------------------------------------------------------


class PatchEmbed3D(nn.Module):
    """Non-overlapping 3D patch embedding.

    Divides a 3D volume into non-overlapping patches, then projects each
    flattened patch to an embedding vector using a 3D convolution.
    This is equivalent to a linear projection of flattened patches.

    Args:
        volume_size: Input volume dimensions (X, Y, Z).
        patch_size: Patch dimensions (Px, Py, Pz).
        in_channels: Number of input channels.
        embed_dim: Output embedding dimension.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        in_channels: int,
        embed_dim: int,
    ) -> None:
        super().__init__()
        self.volume_size = volume_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        for i in range(3):
            assert volume_size[i] % patch_size[i] == 0, (
                f"volume_size[{i}]={volume_size[i]} must be divisible by patch_size[{i}]={patch_size[i]}"
            )

        self.grid_size: Tuple[int, int, int] = (
            volume_size[0] // patch_size[0],
            volume_size[1] // patch_size[1],
            volume_size[2] // patch_size[2],
        )
        self.num_patches: int = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        # 3D convolution with kernel_size=patch_size and stride=patch_size
        # equivalent to a per-patch linear projection
        self.proj = nn.Conv3d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Patchify and embed a 3D volume.

        Args:
            x: Input volume of shape (B, C, X, Y, Z).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        x = self.proj(x)  # (B, embed_dim, gX, gY, gZ)
        x = rearrange(x, "b e gx gy gz -> b (gx gy gz) e")
        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# Vision Transformer 3D base
# ---------------------------------------------------------------------------


class VisionTransformer3D(nn.Module):
    """3D Vision Transformer.

    A full 3D ViT backbone consisting of patch embedding, positional encoding,
    and a stack of Transformer blocks.  Both context and target encoders are
    instances of this class (target is an EMA copy of context).

    Args:
        volume_size: Input volume dimensions (X, Y, Z).
        patch_size: Patch dimensions (Px, Py, Pz).
        in_channels: Number of input channels.
        config: Encoder configuration dataclass.
    """

    def __init__(
        self,
        volume_size: Tuple[int, int, int],
        patch_size: Tuple[int, int, int],
        in_channels: int,
        config: EncoderConfig,
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed3D(
            volume_size=volume_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=config.embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.embed_dim = config.embed_dim

        # Fixed sinusoidal positional embeddings (not learnable, shape (1, N, D))
        pos_embed = build_3d_sincos_pos_embed(
            embed_dim=config.embed_dim,
            grid_size=self.patch_embed.grid_size,
        )
        self.register_buffer("pos_embed", pos_embed)  # (1, N, D)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=config.embed_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following ViT recommendations."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Process all patches (used by target encoder).

        Args:
            x: Input volume of shape (B, C, X, Y, Z).

        Returns:
            Per-patch representations of shape (B, num_patches, embed_dim).
        """
        tokens = self.patch_embed(x)  # (B, N, D)
        tokens = tokens + self.pos_embed  # add positional encoding
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens

    def forward_masked(
        self, x: torch.Tensor, keep_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process only the visible (context) patches (used by context encoder).

        Args:
            x: Input volume of shape (B, C, X, Y, Z).
            keep_indices: Long tensor of shape (B, n_keep) containing the patch
                indices to keep (0-based, up to num_patches-1).

        Returns:
            Tuple of:
                - context_tokens: Shape (B, n_keep, embed_dim)
                - pos_embed_keep: Positional embeddings for kept patches,
                  shape (B, n_keep, embed_dim). Returned so the predictor
                  can later look up target-position embeddings.
        """
        B = x.shape[0]
        n_keep = keep_indices.shape[1]

        tokens = self.patch_embed(x)  # (B, N, D)

        # Gather only the kept patches (with their positional encodings)
        # keep_indices: (B, n_keep) → expand to (B, n_keep, D) for gather
        idx_expand = keep_indices.unsqueeze(-1).expand(B, n_keep, self.embed_dim)
        pos_expand = self.pos_embed.expand(B, -1, -1)  # (B, N, D)

        tokens_keep = torch.gather(tokens, dim=1, index=idx_expand)  # (B, n_keep, D)
        pos_keep = torch.gather(pos_expand, dim=1, index=idx_expand)  # (B, n_keep, D)

        tokens_keep = tokens_keep + pos_keep  # add positional encoding

        for block in self.blocks:
            tokens_keep = block(tokens_keep)
        tokens_keep = self.norm(tokens_keep)

        return tokens_keep, pos_keep


# ---------------------------------------------------------------------------
# JEPA Predictor
# ---------------------------------------------------------------------------


class JEPAPredictor(nn.Module):
    """JEPA Predictor network.

    Given context encoder outputs, the predictor infers the target encoder
    representations at the masked (target) positions.  It does so by:

    1. Projecting context tokens to the predictor dimension.
    2. Appending learnable mask tokens at each target position,
       conditioned on the target patches' positional embeddings.
    3. Passing the full sequence (context + mask tokens) through a narrow
       Transformer so mask tokens can attend to context.
    4. Projecting outputs at target positions back to the encoder dimension.

    This narrow Transformer design follows the original I-JEPA paper:
    the predictor is intentionally weaker than the encoder to prevent
    trivial solutions.

    Args:
        encoder_dim: Embedding dimension of the encoder.
        config: Predictor configuration dataclass.
    """

    def __init__(self, encoder_dim: int, config: PredictorConfig) -> None:
        super().__init__()
        pred_dim = config.hidden_dim

        # Project from encoder space to predictor space
        self.input_proj = nn.Linear(encoder_dim, pred_dim)

        # Learnable mask token (shared across all target positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))

        # Projection for positional embedding (from encoder dim to pred dim)
        self.pos_proj = nn.Linear(encoder_dim, pred_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=pred_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(pred_dim)

        # Project back to encoder space for loss computation
        self.output_proj = nn.Linear(pred_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        context_tokens: torch.Tensor,
        context_pos: torch.Tensor,
        target_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Predict target encoder representations from context.

        Args:
            context_tokens: Context encoder outputs, shape (B, n_ctx, encoder_dim).
            context_pos: Positional embeddings for context patches,
                shape (B, n_ctx, encoder_dim).
            target_pos: Positional embeddings for target patches,
                shape (B, n_tgt, encoder_dim).

        Returns:
            Predicted representations at target positions,
            shape (B, n_tgt, encoder_dim).
        """
        B, n_ctx = context_tokens.shape[:2]
        n_tgt = target_pos.shape[1]

        # Project context tokens to predictor space and add positional info
        ctx = self.input_proj(context_tokens) + self.pos_proj(context_pos)  # (B, n_ctx, pred_dim)

        # Build mask tokens at target positions
        tgt = self.mask_token.expand(B, n_tgt, -1) + self.pos_proj(target_pos)  # (B, n_tgt, pred_dim)

        # Concatenate: context first, then target queries
        tokens = torch.cat([ctx, tgt], dim=1)  # (B, n_ctx+n_tgt, pred_dim)

        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)

        # Extract only the target query outputs and project back to encoder space
        pred = self.output_proj(tokens[:, n_ctx:, :])  # (B, n_tgt, encoder_dim)
        return pred


# ---------------------------------------------------------------------------
# Complete I-JEPA Model
# ---------------------------------------------------------------------------


class IJEPAModel(nn.Module):
    """Complete I-JEPA model.

    Combines context encoder, target encoder (EMA copy), and predictor.

    The forward pass:
    1. The context encoder processes visible patches.
    2. The target encoder (no gradient) processes ALL patches.
    3. The predictor predicts the target encoder output at target positions.
    4. VICReg loss is applied between predictions and targets.

    Target encoder is updated via EMA after each training step (externally).

    Args:
        config: IJEPAConfig dataclass.
    """

    def __init__(self, config: IJEPAConfig) -> None:
        super().__init__()
        self.config = config

        # Context encoder (trained via backprop)
        self.context_encoder = VisionTransformer3D(
            volume_size=config.volume_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            config=config.encoder,
        )

        # Target encoder (updated via EMA; gradients are never computed)
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Predictor
        self.predictor = JEPAPredictor(
            encoder_dim=config.encoder.embed_dim,
            config=config.predictor,
        )

    @property
    def num_patches(self) -> int:
        return self.context_encoder.num_patches

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """Exponential Moving Average (EMA) update of the target encoder.

        Must be called after each optimizer step.

        Args:
            momentum: EMA decay coefficient (e.g. 0.996).
                Higher values = slower target encoder update.
        """
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_tgt.data.mul_(momentum).add_(param_ctx.data * (1.0 - momentum))

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Full I-JEPA forward pass.

        Args:
            x: Input 3D volume of shape (B, C, X, Y, Z).
            context_indices: Patch indices to use as context,
                shape (B, n_ctx).  Values are in [0, num_patches).
            target_indices_list: List of K tensors, each of shape (B, n_tgt_k)
                containing patch indices for one target block.

        Returns:
            predictions: List of K tensors, each (B, n_tgt_k, embed_dim),
                the predictor's output for each target block.
            targets: List of K tensors, each (B, n_tgt_k, embed_dim),
                the (stop-gradient) target encoder outputs at target positions.
        """
        embed_dim = self.context_encoder.embed_dim
        B = x.shape[0]

        # --- Context encoder (with gradients) ---
        ctx_tokens, ctx_pos = self.context_encoder.forward_masked(x, context_indices)
        # ctx_tokens: (B, n_ctx, D), ctx_pos: (B, n_ctx, D)

        # --- Target encoder (no gradients) ---
        with torch.no_grad():
            all_tokens = self.target_encoder.forward_full(x)  # (B, N, D)

        # Prepare full positional embedding (for looking up target positions)
        full_pos = self.context_encoder.pos_embed.expand(B, -1, -1)  # (B, N, D)

        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for tgt_indices in target_indices_list:
            n_tgt = tgt_indices.shape[1]
            idx_expand = tgt_indices.unsqueeze(-1).expand(B, n_tgt, embed_dim)

            # Target: gather target encoder outputs at target positions
            tgt_tokens = torch.gather(all_tokens, dim=1, index=idx_expand)  # (B, n_tgt, D)
            tgt_pos = torch.gather(full_pos, dim=1, index=idx_expand)  # (B, n_tgt, D)

            # Predictor: predict target from context
            pred = self.predictor(ctx_tokens, ctx_pos, tgt_pos)  # (B, n_tgt, D)

            predictions.append(pred)
            targets.append(tgt_tokens.detach())  # stop gradient to targets

        return predictions, targets


# ---------------------------------------------------------------------------
# 3D-JEPA model (CNN-based volumetric encoder)
# ---------------------------------------------------------------------------


class ConvBlock3D(nn.Module):
    """3D Convolutional block: Conv3d → GroupNorm → GELU, with residual."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        # Find the largest divisor of out_channels that is <= 32
        num_groups = max(g for g in range(1, min(32, out_channels) + 1) if out_channels % g == 0)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(num_groups, out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.GELU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(num_groups, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class JEPA3DEncoder(nn.Module):
    """CNN-based 3D encoder for 3D-JEPA.

    Hierarchical 3D CNN that extracts multi-scale volumetric features,
    then maps them to per-patch representations via patch pooling.

    Args:
        in_channels: Number of input channels.
        embed_dim: Output embedding dimension per patch.
        patch_size: Non-overlapping patch size (Px, Py, Pz).
    """

    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 512,
        patch_size: Tuple[int, int, int] = (16, 16, 16),
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Hierarchical feature extraction: downsample 4x in each spatial dim
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.layer1 = ConvBlock3D(64, 128, stride=2)   # /2
        self.layer2 = ConvBlock3D(128, 256, stride=2)  # /4
        self.layer3 = ConvBlock3D(256, embed_dim, stride=1)  # same spatial

        # Pooling to patch resolution
        # After 4x downsampling, if patch_size=16 we get 16/4 = 4 → pool(4)
        self._pool_factor = patch_size[0] // 4
        assert patch_size[0] == patch_size[1] == patch_size[2], (
            "JEPA3DEncoder requires isotropic patch sizes."
        )
        self.patch_pool = nn.AdaptiveAvgPool3d(1) if self._pool_factor == 1 else None
        if self._pool_factor > 1:
            self.patch_pool = nn.AvgPool3d(kernel_size=self._pool_factor)

        self.proj_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract per-patch volumetric features.

        Args:
            x: Input volume of shape (B, C, X, Y, Z).

        Returns:
            Per-patch features of shape (B, num_patches, embed_dim).
        """
        h = self.stem(x)
        h = self.layer1(h)   # (B, 128, X/2, Y/2, Z/2)
        h = self.layer2(h)   # (B, 256, X/4, Y/4, Z/4)
        h = self.layer3(h)   # (B, embed_dim, X/4, Y/4, Z/4)
        h = self.patch_pool(h)  # (B, embed_dim, gX, gY, gZ)
        h = rearrange(h, "b e gx gy gz -> b (gx gy gz) e")
        h = self.proj_norm(h)
        return h


class JEPA3DModel(nn.Module):
    """3D-JEPA model with CNN-based volumetric encoder.

    Uses a hierarchical 3D CNN encoder instead of a Vision Transformer,
    making it more suited to the spatial coherence structure of 3D medical
    imaging while still following the JEPA prediction objective.

    Args:
        config: JEPA3DConfig dataclass.
    """

    def __init__(self, config: JEPA3DConfig) -> None:
        super().__init__()
        self.config = config

        # Context and target encoders (CNN-based)
        self.context_encoder = JEPA3DEncoder(
            in_channels=config.in_channels,
            embed_dim=config.encoder.embed_dim,
            patch_size=config.patch_size,
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad_(False)

        # Predictor
        self.predictor = JEPAPredictor(
            encoder_dim=config.encoder.embed_dim,
            config=config.predictor,
        )

        # 3D positional embeddings for the predictor
        px, py, pz = config.patch_size
        grid_size = (
            config.volume_size[0] // px,
            config.volume_size[1] // py,
            config.volume_size[2] // pz,
        )
        pos_embed = build_3d_sincos_pos_embed(config.encoder.embed_dim, grid_size)
        self.register_buffer("pos_embed", pos_embed)  # (1, N, embed_dim)
        self.num_patches = grid_size[0] * grid_size[1] * grid_size[2]

    @torch.no_grad()
    def update_target_encoder(self, momentum: float) -> None:
        """EMA update of the CNN target encoder.

        Args:
            momentum: EMA decay coefficient.
        """
        for param_ctx, param_tgt in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            param_tgt.data.mul_(momentum).add_(param_ctx.data * (1.0 - momentum))

    def _gather_patches(
        self, all_tokens: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        """Helper: gather tokens at specified patch indices.

        Args:
            all_tokens: Shape (B, N, D).
            indices: Shape (B, n) with values in [0, N).

        Returns:
            Gathered tokens, shape (B, n, D).
        """
        B, n = indices.shape
        D = all_tokens.shape[-1]
        idx_expand = indices.unsqueeze(-1).expand(B, n, D)
        return torch.gather(all_tokens, dim=1, index=idx_expand)

    def _mask_context(
        self, all_tokens: torch.Tensor, context_indices: torch.Tensor
    ) -> torch.Tensor:
        """Select context tokens by index.

        Args:
            all_tokens: Shape (B, N, D).
            context_indices: Shape (B, n_ctx).

        Returns:
            Context tokens of shape (B, n_ctx, D).
        """
        return self._gather_patches(all_tokens, context_indices)

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices_list: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """3D-JEPA forward pass (mirrors IJEPAModel.forward interface).

        Args:
            x: Input 3D volume of shape (B, C, X, Y, Z).
            context_indices: Patch indices to use as context, shape (B, n_ctx).
            target_indices_list: List of K tensors, each (B, n_tgt_k).

        Returns:
            predictions: List of K tensors, each (B, n_tgt_k, embed_dim).
            targets: List of K tensors, each (B, n_tgt_k, embed_dim).
        """
        B = x.shape[0]
        embed_dim = self.config.encoder.embed_dim

        # Context encoder (with gradients): only process context patches' features
        all_ctx = self.context_encoder(x)  # (B, N, D)
        ctx_tokens = self._mask_context(all_ctx, context_indices)  # (B, n_ctx, D)
        ctx_pos = self._gather_patches(
            self.pos_embed.expand(B, -1, -1), context_indices
        )

        # Target encoder (no gradients)
        with torch.no_grad():
            all_tgt = self.target_encoder(x)  # (B, N, D)

        full_pos = self.pos_embed.expand(B, -1, -1)  # (B, N, D)

        predictions: List[torch.Tensor] = []
        targets: List[torch.Tensor] = []

        for tgt_indices in target_indices_list:
            n_tgt = tgt_indices.shape[1]
            tgt_tokens = self._gather_patches(all_tgt, tgt_indices)  # (B, n_tgt, D)
            tgt_pos = self._gather_patches(full_pos, tgt_indices)  # (B, n_tgt, D)

            pred = self.predictor(ctx_tokens, ctx_pos, tgt_pos)  # (B, n_tgt, D)
            predictions.append(pred)
            targets.append(tgt_tokens.detach())

        return predictions, targets
