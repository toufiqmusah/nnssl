"""
Brain-JEPA architecture module for nnssl.

Implements the Joint Embedding Predictive Architecture (JEPA) for 3D medical
imaging using Primus-style EVA encoders.  Closely follows the module convention
established by ``evaMAE_module.py``.

Three public classes:

* :class:`PrimusMultiStageEncoder` — Patch-embed → EVA with intermediate
  feature extraction at configurable layer indices.
* :class:`BrainJEPAPredictor` — Narrow transformer that maps context features
  + positional queries → predicted target features (one per stage).
* :class:`BrainJEPA` — Top-level module composing context encoder, EMA target
  encoder, and stage-wise predictors.  ``forward()`` returns
  ``(loss, info_dict)`` directly.

Reference:
    Assran et al., "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture" (I-JEPA), CVPR 2023.
"""

from __future__ import annotations

import copy
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from dynamic_network_architectures.building_blocks.eva import Eva
from dynamic_network_architectures.building_blocks.patch_encode_decode import PatchEmbed
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


# ============================================================================ #
#  Multi-stage encoder (wraps PatchEmbed + EVA)
# ============================================================================ #


class PrimusMultiStageEncoder(nn.Module):
    """Primus-style encoder that extracts features at multiple EVA depths.

    This is essentially the *encoder half* of ``EvaMAE``, augmented so that
    intermediate representations can be tapped.  It does **not** include a
    decoder or pixel prediction head.

    Parameters
    ----------
    input_channels : int
    embed_dim : int
    patch_embed_size : tuple of int
        Voxel-level patch size, e.g. ``(8, 8, 8)``.
    eva_depth : int
    eva_numheads : int
    input_shape : tuple of int
        Spatial shape of the input volume, e.g. ``(160, 160, 160)``.
    stage_indices : list of int or None
        Layer indices (0-based) at which to capture features.  ``None`` →
        auto-compute as thirds of ``eva_depth``.
    drop_path_rate : float
    init_values : float or None
    scale_attn_inner : bool
    """

    def __init__(
        self,
        input_channels: int,
        embed_dim: int,
        patch_embed_size: Tuple[int, ...],
        eva_depth: int,
        eva_numheads: int,
        input_shape: Tuple[int, ...],
        stage_indices: list[int] | None = None,
        drop_path_rate: float = 0.2,
        init_values: float | None = 0.1,
        scale_attn_inner: bool = True,
    ):
        super().__init__()
        assert len(input_shape) == 3, "Only 3D inputs supported"
        assert all(s % p == 0 for s, p in zip(input_shape, patch_embed_size))

        self.patch_embed_size = patch_embed_size
        self.embed_dim = embed_dim
        self.patch_grid = tuple(s // p for s, p in zip(input_shape, patch_embed_size))

        # --- Patch embedding (same as EvaMAE) ---
        self.down_projection = PatchEmbed(patch_embed_size, input_channels, embed_dim)
        self.down_projection.apply(InitWeights_He(1e-2))

        # --- EVA transformer ---
        self.eva = Eva(
            embed_dim=embed_dim,
            depth=eva_depth,
            num_heads=eva_numheads,
            ref_feat_shape=self.patch_grid,
            num_reg_tokens=0,
            use_rot_pos_emb=True,
            use_abs_pos_emb=True,
            mlp_ratio=4 * 2 / 3,
            drop_path_rate=drop_path_rate,
            patch_drop_rate=0.0,  # no random token dropping in JEPA
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            init_values=init_values,
            scale_attn_inner=scale_attn_inner,
        )

        # --- Stage extraction indices ---
        if stage_indices is None:
            third = max(1, eva_depth // 3)
            stage_indices = [third - 1, 2 * third - 1, eva_depth - 1]
        self.stage_indices: list[int] = sorted(set(stage_indices))
        self.num_stages = len(self.stage_indices)

        # Register hooks to capture intermediate features
        self._stage_features: list[torch.Tensor] = []
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._register_stage_hooks()

    # ------------------------------------------------------------------ #
    #  Hook management
    # ------------------------------------------------------------------ #

    def _register_stage_hooks(self):
        """Attach forward hooks to EVA blocks at ``stage_indices``."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

        for idx in self.stage_indices:
            block = self.eva.blocks[idx]
            hook = block.register_forward_hook(self._capture_hook)
            self._hooks.append(hook)

    def _capture_hook(self, module, input, output):
        self._stage_features.append(output)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor  [B, C, D, H, W]

        Returns
        -------
        stage_features : list of Tensor  [B, N, embed_dim]
            One per stage index.
        """
        self._stage_features = []

        # Patch embed → [B, embed_dim, Gd, Gh, Gw]
        x = self.down_projection(x)
        x = rearrange(x, "b c d h w -> b (d h w) c")

        # EVA forward (hooks capture intermediate features)
        x, _ = self.eva(x)

        # The hooks may have captured features; last block's output is `x`
        # but it was also captured by the hook.
        features = list(self._stage_features)
        self._stage_features = []

        assert len(features) == self.num_stages, (
            f"Expected {self.num_stages} stage features, got {len(features)}. "
            f"stage_indices={self.stage_indices}, eva_depth={len(self.eva.blocks)}"
        )
        return features


# ============================================================================ #
#  JEPA Predictor
# ============================================================================ #


class BrainJEPAPredictor(nn.Module):
    """Narrow transformer predictor for one stage of JEPA.

    Takes context patch features and positional embeddings for target locations,
    then predicts the target encoder's features at those locations.

    Parameters
    ----------
    context_dim : int
        Dimension of the context encoder's features (``embed_dim``).
    pred_dim : int
        Internal predictor dimension.
    pred_depth : int
        Number of transformer blocks.
    pred_heads : int
        Number of attention heads.
    num_patches : int
        Total number of patches ``N = prod(patch_grid)`` (for learned pos embed).
    """

    def __init__(
        self,
        context_dim: int,
        pred_dim: int,
        pred_depth: int,
        pred_heads: int,
        num_patches: int,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.pred_dim = pred_dim

        # Project context features to predictor dimension
        self.context_proj = nn.Linear(context_dim, pred_dim)

        # Learnable positional embeddings for all patch positions
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, pred_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Mask token — query for target positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=pred_dim,
                    nhead=pred_heads,
                    dim_feedforward=pred_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(pred_depth)
            ]
        )
        self.norm = nn.LayerNorm(pred_dim)

        # Project back to context_dim for loss computation
        self.output_proj = nn.Linear(pred_dim, context_dim)

    def forward(
        self,
        context_features: torch.Tensor,
        context_mask: torch.BoolTensor,
        target_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        context_features : Tensor  [B, N, D_ctx]
            Full-grid features from the context encoder (only context positions
            are meaningful, but all positions are present).
        context_mask : BoolTensor  [B, N]
            True for context (visible) positions.
        target_mask : BoolTensor  [B, N]
            True for target positions to predict.

        Returns
        -------
        predictions : Tensor  [B, N_target, D_ctx]
            Predicted features at target positions.
        """
        B, N, _ = context_features.shape

        # 1. Project context features to predictor dim
        ctx = self.context_proj(context_features)  # [B, N, pred_dim]

        # 2. Build query sequence: context tokens + mask tokens at target positions
        #    We construct a full-grid sequence of pred_dim, then select context + target.
        mask_tokens = self.mask_token.expand(B, N, -1)  # [B, N, pred_dim]

        # Merge: use projected context at context positions, mask tokens at target
        tokens = mask_tokens.clone()
        ctx_expanded = context_mask.unsqueeze(-1).expand_as(tokens)
        tokens = torch.where(ctx_expanded, ctx, tokens)

        # Add positional embeddings
        tokens = tokens + self.pos_embed

        # 3. Select only context + target tokens for efficiency
        relevant = context_mask | target_mask  # [B, N]

        # Process per-sample because each sample may have different numbers
        predictions_list = []
        for b in range(B):
            rel_idx = relevant[b].nonzero(as_tuple=True)[0]  # [K]
            tgt_idx = target_mask[b].nonzero(as_tuple=True)[0]  # [K_tgt]

            x = tokens[b:b+1, rel_idx]  # [1, K, pred_dim]
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)

            # Map rel_idx back to find which positions in x correspond to targets
            # Build a lookup: position → index in rel_idx
            rel_idx_set = rel_idx.tolist()
            tgt_local = torch.tensor(
                [rel_idx_set.index(t.item()) for t in tgt_idx],
                device=x.device,
            )
            pred_tokens = x[0, tgt_local]  # [K_tgt, pred_dim]
            predictions_list.append(pred_tokens)

        # Pad to same length across batch (max target count)
        max_tgt = max(p.shape[0] for p in predictions_list)
        padded = torch.zeros(B, max_tgt, self.pred_dim, device=context_features.device)
        tgt_counts = []
        for b, p in enumerate(predictions_list):
            padded[b, : p.shape[0]] = p
            tgt_counts.append(p.shape[0])

        # Project to context_dim
        output = self.output_proj(padded)  # [B, max_tgt, D_ctx]
        return output, tgt_counts


# ============================================================================ #
#  BrainJEPA — top-level module
# ============================================================================ #


class BrainJEPA(nn.Module):
    """Joint Embedding Predictive Architecture for 3D brain imaging.

    Composes a trainable context encoder, an EMA-updated target encoder,
    and stage-wise predictors.  ``forward()`` returns ``(loss, info_dict)``
    directly — the loss is smooth-L1 in feature space, not pixel space.

    Parameters
    ----------
    primus_kwargs : dict
        Keyword arguments forwarded to :class:`PrimusMultiStageEncoder`.
    stage_indices : list of int or None
        Passed to the encoder; ``None`` → auto thirds of ``eva_depth``.
    pred_dim, pred_depth, pred_heads : int
        Predictor architecture dimensions.
    ema_decay_start, ema_decay_end : float
        Cosine-annealed EMA decay schedule bounds.
    stage_loss_weights : list of float or None
        Per-stage loss weights.  ``None`` → ``[0.5, 0.75, 1.0]``.
    """

    def __init__(
        self,
        primus_kwargs: dict,
        stage_indices: list[int] | None = None,
        pred_dim: int = 384,
        pred_depth: int = 4,
        pred_heads: int = 6,
        ema_decay_start: float = 0.996,
        ema_decay_end: float = 1.0,
        stage_loss_weights: list[float] | None = None,
    ):
        super().__init__()

        # ------------------------------------------------------------------ #
        #  nnssl downstream-compatibility attributes  (§3 of design doc)
        # ------------------------------------------------------------------ #
        self.key_to_encoder = "context_encoder.eva"
        self.key_to_stem = "context_encoder.down_projection"
        self.keys_to_in_proj = ("context_encoder.down_projection.proj",)
        self.key_to_lpe = "context_encoder.eva.pos_embed"

        # ------------------------------------------------------------------ #
        #  Context encoder (trainable)
        # ------------------------------------------------------------------ #
        self.context_encoder = PrimusMultiStageEncoder(
            stage_indices=stage_indices, **primus_kwargs
        )

        # ------------------------------------------------------------------ #
        #  Target encoder (EMA copy, frozen)
        # ------------------------------------------------------------------ #
        self.target_encoder = PrimusMultiStageEncoder(
            stage_indices=stage_indices, **primus_kwargs
        )
        # Copy weights and freeze
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------ #
        #  Predictors (one per stage)
        # ------------------------------------------------------------------ #
        num_stages = self.context_encoder.num_stages
        embed_dim = self.context_encoder.embed_dim
        num_patches = self.context_encoder.patch_grid[0] * \
                      self.context_encoder.patch_grid[1] * \
                      self.context_encoder.patch_grid[2]

        self.predictors = nn.ModuleList(
            [
                BrainJEPAPredictor(
                    context_dim=embed_dim,
                    pred_dim=pred_dim,
                    pred_depth=pred_depth,
                    pred_heads=pred_heads,
                    num_patches=num_patches,
                )
                for _ in range(num_stages)
            ]
        )

        # ------------------------------------------------------------------ #
        #  Loss weights
        # ------------------------------------------------------------------ #
        if stage_loss_weights is None:
            stage_loss_weights = [0.5, 0.75, 1.0]
        # Pad or trim to match num_stages
        while len(stage_loss_weights) < num_stages:
            stage_loss_weights.append(1.0)
        self.stage_loss_weights = stage_loss_weights[:num_stages]

        # EMA schedule parameters
        self.ema_decay_start = ema_decay_start
        self.ema_decay_end = ema_decay_end

    # ------------------------------------------------------------------ #
    #  EMA update
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def update_target_encoder(self, step: int = 0, total_steps: int = 1):
        """Cosine-annealed EMA update of the target encoder (§3)."""
        tau = self.ema_decay_end - (self.ema_decay_end - self.ema_decay_start) * (
            math.cos(math.pi * step / total_steps) + 1
        ) / 2
        for ctx_p, tgt_p in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters(),
        ):
            tgt_p.data.mul_(tau).add_(ctx_p.data, alpha=1.0 - tau)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x: torch.Tensor,
        context_mask: torch.BoolTensor,
        target_masks: torch.BoolTensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Parameters
        ----------
        x : Tensor  [B, C, D, H, W]
        context_mask : BoolTensor  [B, N]
        target_masks : BoolTensor  [B, T, N]

        Returns
        -------
        loss : Tensor  scalar
        info : dict
            Per-stage losses and cosine similarity metrics.
        """
        B, T, N = target_masks.shape

        # --- Context encoder (trainable) ---
        ctx_features = self.context_encoder(x)  # list of [B, N, D]

        # --- Target encoder (EMA, no grad) ---
        with torch.no_grad():
            tgt_features = self.target_encoder(x)  # list of [B, N, D]

        # --- Per-stage prediction and loss ---
        total_loss = torch.tensor(0.0, device=x.device)
        info: dict[str, float] = {}

        for s, (ctx_feat, tgt_feat, predictor) in enumerate(
            zip(ctx_features, tgt_features, self.predictors)
        ):
            stage_loss = torch.tensor(0.0, device=x.device)

            for t in range(T):
                tgt_mask_t = target_masks[:, t]  # [B, N]

                # Predict target features from context
                pred, tgt_counts = predictor(ctx_feat, context_mask, tgt_mask_t)
                # [B, max_tgt, D]

                # Gather actual target features
                tgt_list = []
                for b in range(B):
                    tgt_idx = tgt_mask_t[b].nonzero(as_tuple=True)[0]
                    tgt_list.append(tgt_feat[b, tgt_idx])  # [K_tgt, D]

                # Compute smooth-L1 loss per sample, averaged
                for b in range(B):
                    k = tgt_counts[b]
                    if k > 0:
                        p = pred[b, :k]  # [K, D]
                        t_feat = tgt_list[b].detach()  # [K, D]
                        stage_loss = stage_loss + F.smooth_l1_loss(p, t_feat)

            stage_loss = stage_loss / (B * T)

            # Cosine similarity metric (detached, for monitoring)
            with torch.no_grad():
                cos_sims = []
                for b in range(B):
                    tgt_mask_0 = target_masks[b, 0]
                    tgt_idx = tgt_mask_0.nonzero(as_tuple=True)[0]
                    if len(tgt_idx) > 0:
                        p = pred[b, : len(tgt_idx)]
                        t_f = tgt_feat[b, tgt_idx]
                        cos = F.cosine_similarity(p, t_f, dim=-1).mean()
                        cos_sims.append(cos.item())
                if cos_sims:
                    info[f"cos_sim_stage{s}"] = sum(cos_sims) / len(cos_sims)

            weighted_loss = self.stage_loss_weights[s] * stage_loss
            total_loss = total_loss + weighted_loss
            info[f"loss_stage{s}"] = stage_loss.item()

        return total_loss, info
