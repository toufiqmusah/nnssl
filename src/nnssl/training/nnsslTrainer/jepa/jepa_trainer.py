"""
JEPA (Joint-Embedding Predictive Architecture) trainers for 3D medical image
self-supervised pretraining.

Implements:
- IJEPATrainer: I-JEPA trainer using 3D Vision Transformer.
- JEPA3DTrainer: 3D-JEPA trainer using CNN-based volumetric encoder.
- Convenience variants for different batch sizes.

Both trainers follow the AbstractBaseTrainer interface and integrate with
the nnssl training infrastructure (dataloaders, logging, checkpointing, etc.).

Training algorithm:
1. For each batch of 3D volumes:
   a. Generate block masks (context + K target blocks).
   b. Context encoder processes visible patches → context representations.
   c. Target encoder (EMA) processes all patches → target representations.
   d. Predictor maps context to predicted target representations.
   e. VICReg loss between predictions and (detached) targets.
   f. Backprop through context encoder + predictor only.
   g. EMA-update target encoder from context encoder.

References:
    Assran et al., "Self-Supervised Learning from Images with a
    Joint-Embedding Predictive Architecture", CVPR 2023.
    Bardes et al., "VICReg: Variance-Invariance-Covariance Regularization
    for Self-Supervised Learning", ICLR 2022.
"""

import math
from copy import deepcopy
from typing import List, Tuple, Union

import numpy as np
import torch
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import save_json
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch import autocast, nn
from torch.optim import AdamW

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan, Plan
from nnssl.ssl_data.configure_basic_dummyDA import (
    configure_rotation_dummyDA_mirroring_and_inital_patch_size,
)
from nnssl.ssl_data.data_augmentation.jepa_augmentations import (
    JEPAAugmentationPipeline,
    JEPAValidationTransform,
)
from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
from nnssl.training.loss.abstract_loss import AbstractLoss
from nnssl.training.nnsslTrainer.AbstractTrainer import AbstractBaseTrainer
from nnssl.training.nnsslTrainer.jepa.jepa_config import (
    IJEPAConfig,
    JEPA3DConfig,
    VICRegLossConfig,
)
from nnssl.training.nnsslTrainer.jepa.jepa_data import BlockMaskGenerator3D
from nnssl.training.nnsslTrainer.jepa.jepa_loss import VICRegLoss
from nnssl.training.nnsslTrainer.jepa.jepa_models import JEPA3DModel, IJEPAModel
from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnssl.utilities.helpers import dummy_context


# ---------------------------------------------------------------------------
# Helper: cosine-scheduled EMA momentum
# ---------------------------------------------------------------------------


def cosine_ema_momentum(
    start: float,
    end: float,
    current_step: int,
    total_steps: int,
) -> float:
    """Compute EMA momentum following a cosine schedule.

    Momentum increases from ``start`` to ``end`` over ``total_steps``.

    Args:
        start: Initial momentum value (e.g. 0.996).
        end: Final momentum value (e.g. 1.0).
        current_step: Current training step (0-based).
        total_steps: Total number of training steps.

    Returns:
        Momentum value for the current step.
    """
    ratio = current_step / max(total_steps, 1)
    return end - (end - start) * (0.5 * (1.0 + math.cos(math.pi * ratio)))


# ---------------------------------------------------------------------------
# Base JEPA Trainer (shared logic)
# ---------------------------------------------------------------------------


class _BaseJEPATrainer(AbstractBaseTrainer):
    """Shared JEPA trainer logic.

    Both I-JEPA and 3D-JEPA trainers inherit from this class.  It provides:
    - EMA momentum schedule computation.
    - Overridden ``initialize()`` that skips the architecture verification
      step (not needed for custom ViT/CNN architectures).
    - Overridden ``configure_optimizers()`` with AdamW + LinearWarmupCosine.
    - Shared ``train_step`` and ``validation_step`` implementations.
    - Logging of variance/invariance/covariance loss components.
    """

    # Subclasses set these:
    _jepa_config: Union[IJEPAConfig, JEPA3DConfig]
    _model_class: type  # IJEPAModel or JEPA3DModel

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
        jepa_config: Union[IJEPAConfig, JEPA3DConfig, None] = None,
    ) -> None:
        # Set patch size from JEPA config before calling super (which saves config_plan)
        jepa_cfg = jepa_config or self._get_default_jepa_config()
        plan.configurations[configuration_name].patch_size = jepa_cfg.volume_size
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        self.jepa_config = jepa_cfg
        self._total_training_steps = self.num_epochs * self.num_iterations_per_epoch
        self._global_step: int = 0

        # Block mask generator — created after patch size is known
        volume_size = self.jepa_config.volume_size
        patch_size = self.jepa_config.patch_size
        grid_size: Tuple[int, int, int] = (
            volume_size[0] // patch_size[0],
            volume_size[1] // patch_size[1],
            volume_size[2] // patch_size[2],
        )
        self._mask_generator = BlockMaskGenerator3D(
            grid_size=grid_size,
            masking_config=self.jepa_config.masking,
        )

    def _get_default_jepa_config(self) -> Union[IJEPAConfig, JEPA3DConfig]:
        raise NotImplementedError("Subclasses must implement _get_default_jepa_config.")

    # ------------------------------------------------------------------
    # Override initialize() to skip unsupported verify_adaptation_plans
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Initialize JEPA trainer.

        Overrides the base ``initialize()`` to skip the architecture
        verification step, which is not compatible with custom ViT / CNN
        architectures not registered in the nnssl architecture registry.
        """
        if not self.was_initialized:
            self._set_batch_size()
            self.network, self.adaptation_plan = self.build_architecture_and_adaptation_plan(
                self.config_plan, self.num_input_channels, self.num_output_channels
            )
            save_json(self.adaptation_plan.serialize(), self.adaptation_json_plan)
            self.network.to(self.device)
            # NOTE: verify_adaptation_plans is intentionally skipped here because
            # the JEPA ViT/CNN architectures are not registered in the nnssl
            # architecture registry used by the verification step.

            self.optimizer, self.lr_scheduler = self.configure_optimizers()

            if self.is_ddp:
                from torch.nn.parallel import DistributedDataParallel as DDP

                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(
                    self.network,
                    device_ids=[self.local_rank],
                    find_unused_parameters=True,
                )

            self.loss = self.build_loss()
            self.was_initialized = True
        else:
            raise RuntimeError(
                "You have called self.initialize even though the trainer was already initialized. "
                "That should not happen."
            )

    # ------------------------------------------------------------------
    # Optimizer: AdamW + warm-up cosine annealing
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        """Configure AdamW with linear warm-up cosine annealing LR schedule.

        Only the context encoder and predictor parameters are optimised.
        The target encoder is updated via EMA and must not receive gradients.
        """
        # Unwrap DDP if necessary
        net = self.network.module if self.is_ddp else self.network
        trainable_params = list(net.context_encoder.parameters()) + list(
            net.predictor.parameters()
        )
        optimizer = AdamW(
            trainable_params,
            lr=self.initial_lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=40,
            max_epochs=self.num_epochs,
        )
        return optimizer, lr_scheduler

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def build_loss(self) -> nn.Module:
        return VICRegLoss(config=self.jepa_config.loss)

    # ------------------------------------------------------------------
    # Transforms (shared by both subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def get_training_transforms(
        patch_size,
        rotation_for_DA,
        mirror_axes,
        do_dummy_2d_data_aug,
        order_resampling_data: int = 3,
        order_resampling_seg: int = 1,
        border_val_seg: int = -1,
    ) -> AbstractTransform:
        """Return training augmentation pipeline.

        Applies mild spatial and intensity augmentations followed by
        NumpyToTensor conversion.  Masking is performed inside ``train_step``
        rather than in the data pipeline.
        """
        return JEPAAugmentationPipeline(
            do_spatial_aug=True,
            do_intensity_aug=True,
            to_tensor=True,
        )

    @staticmethod
    def get_validation_transforms() -> AbstractTransform:
        """Return validation transforms — tensor conversion only."""
        return JEPAValidationTransform(data_key="data")

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def get_dataloaders(self):
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
        )
        val_transforms = self.get_validation_transforms()

        dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, tr_transforms)
            mt_gen_val = SingleThreadedAugmenter(dl_val, val_transforms)
        else:
            mt_gen_train = LimitedLenWrapper(
                self.num_iterations_per_epoch,
                data_loader=dl_tr,
                transform=tr_transforms,
                num_processes=allowed_num_processes,
                num_cached=6,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
            mt_gen_val = LimitedLenWrapper(
                self.num_val_iterations_per_epoch,
                data_loader=dl_val,
                transform=val_transforms,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=3,
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.02,
            )
        return mt_gen_train, mt_gen_val

    # ------------------------------------------------------------------
    # EMA momentum
    # ------------------------------------------------------------------

    @property
    def current_momentum(self) -> float:
        """Current EMA momentum value (cosine-scheduled or constant)."""
        if self.jepa_config.use_momentum_schedule:
            return cosine_ema_momentum(
                start=self.jepa_config.momentum_start,
                end=self.jepa_config.momentum_end,
                current_step=self._global_step,
                total_steps=self._total_training_steps,
            )
        return self.jepa_config.momentum_start

    # ------------------------------------------------------------------
    # Shared train / validation step logic
    # ------------------------------------------------------------------

    def _forward_pass(
        self, data: torch.Tensor, is_train: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared forward pass used by both train and validation steps.

        Args:
            data: Input 3D volume tensor of shape (B, C, X, Y, Z).
            is_train: Whether to compute gradients (True) or not (False).

        Returns:
            Tuple of (total_loss, loss_var, loss_inv, loss_cov) as scalars.
        """
        # Generate block masks on the CPU (lightweight)
        context_indices, target_indices_list = self._mask_generator(data.shape[0])
        context_indices = context_indices.to(self.device, non_blocking=True)
        target_indices_list = [t.to(self.device, non_blocking=True) for t in target_indices_list]

        net = self.network.module if self.is_ddp else self.network

        ctx_mgr = (
            (autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context())
            if is_train
            else torch.no_grad()
        )

        with ctx_mgr:
            predictions, targets = net(data, context_indices, target_indices_list)

        # Aggregate predictions and targets from all K target blocks
        # Each prediction/target has shape (B, n_tgt_k, D)
        all_pred = torch.cat(predictions, dim=1)  # (B, sum_k n_tgt_k, D)
        all_tgt = torch.cat(targets, dim=1)

        # VICReg loss over flattened (B * n_tgt_total, D) tensors
        total_loss, loss_var, loss_inv, loss_cov = self.loss(all_pred, all_tgt)
        return total_loss, loss_var, loss_inv, loss_cov

    def train_step(self, batch: dict) -> dict:
        """Single JEPA training step.

        1. Forward pass through context + target encoders and predictor.
        2. Compute VICReg loss.
        3. Backpropagate through context encoder and predictor.
        4. EMA-update the target encoder.

        Args:
            batch: Data dictionary with key ``"data"``, shape (B, C, X, Y, Z).

        Returns:
            Dictionary with loss components for logging:
                ``loss``, ``loss_var``, ``loss_inv``, ``loss_cov``.
        """
        data: torch.Tensor = batch["data"].to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        total_loss, loss_var, loss_inv, loss_cov = self._forward_pass(data, is_train=True)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(total_loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad], 1.0
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad], 1.0
            )
            self.optimizer.step()

        # EMA update of target encoder (no gradient)
        net = self.network.module if self.is_ddp else self.network
        net.update_target_encoder(self.current_momentum)

        self._global_step += 1

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_var": loss_var.detach().cpu().numpy(),
            "loss_inv": loss_inv.detach().cpu().numpy(),
            "loss_cov": loss_cov.detach().cpu().numpy(),
        }

    def validation_step(self, batch: dict) -> dict:
        """Single JEPA validation step (no gradient, no EMA update).

        Args:
            batch: Data dictionary with key ``"data"``, shape (B, C, X, Y, Z).

        Returns:
            Dictionary with loss components for logging.
        """
        data: torch.Tensor = batch["data"].to(self.device, non_blocking=True)

        with torch.no_grad():
            total_loss, loss_var, loss_inv, loss_cov = self._forward_pass(data, is_train=False)

        return {
            "loss": total_loss.detach().cpu().numpy(),
            "loss_var": loss_var.detach().cpu().numpy(),
            "loss_inv": loss_inv.detach().cpu().numpy(),
            "loss_cov": loss_cov.detach().cpu().numpy(),
        }


# ---------------------------------------------------------------------------
# I-JEPA Trainer (3D Vision Transformer)
# ---------------------------------------------------------------------------


class IJEPATrainer(_BaseJEPATrainer):
    """I-JEPA trainer using a 3D Vision Transformer encoder.

    Implements the I-JEPA (Image JEPA) algorithm for 3D medical images.
    The context encoder and predictor use a ViT-style Transformer;
    the target encoder is an EMA copy of the context encoder.

    Default configuration:
        - Volume size: 96³ voxels
        - Patch size: 16³ voxels → 6×6×6 = 216 patches
        - Encoder: ViT-S scale (embed_dim=384, depth=6, heads=6)
        - Predictor: narrow Transformer (hidden_dim=384, depth=6)
        - Masking: 4 target blocks of 2³ patches each
        - Loss: VICReg with default weights
        - Momentum: 0.996 → 1.0 (cosine schedule)

    Args:
        plan: Pre-training plan.
        configuration_name: Configuration key in the plan.
        fold: Cross-validation fold index.
        pretrain_json: Pre-training dataset metadata.
        device: Compute device.
        jepa_config: Optional custom IJEPAConfig.  Defaults to ``IJEPAConfig()``.
    """

    def _get_default_jepa_config(self) -> IJEPAConfig:
        return IJEPAConfig()

    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> Tuple[nn.Module, AdaptationPlan]:
        """Build the IJEPAModel and create the downstream adaptation plan.

        The adaptation plan points to the context encoder's Transformer blocks
        and patch embedding, enabling downstream fine-tuning with a standard
        ViT-style network.

        Args:
            config_plan: Configuration plan.
            num_input_channels: Number of input channels.
            num_output_channels: Not used (JEPA is self-supervised).

        Returns:
            Tuple of (IJEPAModel, AdaptationPlan).
        """
        model = IJEPAModel(config=self.jepa_config)

        # Build adaptation plan so downstream code knows how to load the encoder
        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans(arch_class_name="ResEncL"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=self.jepa_config.in_channels,
            key_to_encoder="context_encoder.blocks",
            key_to_stem="context_encoder.patch_embed",
            keys_to_in_proj=("context_encoder.patch_embed.proj",),
        )
        return model, adapt_plan


# ---------------------------------------------------------------------------
# 3D-JEPA Trainer (CNN-based volumetric encoder)
# ---------------------------------------------------------------------------


class JEPA3DTrainer(_BaseJEPATrainer):
    """3D-JEPA trainer using a CNN-based volumetric encoder.

    Implements 3D-JEPA with a hierarchical 3D CNN encoder that better
    exploits the continuous volumetric structure of medical images compared
    to a patch-based ViT approach.

    Default configuration:
        - Volume size: 128³ voxels
        - Patch size: 16³ voxels → 8×8×8 = 512 patches
        - Encoder: CNN (embed_dim=512) + 3D AvgPool to patch resolution
        - Predictor: narrow Transformer (hidden_dim=384, depth=6)
        - Masking: 4 target blocks of 3³ patches each
        - Loss: VICReg with default weights
        - Momentum: 0.998 → 1.0 (cosine schedule)

    Args:
        plan: Pre-training plan.
        configuration_name: Configuration key in the plan.
        fold: Cross-validation fold index.
        pretrain_json: Pre-training dataset metadata.
        device: Compute device.
        jepa_config: Optional custom JEPA3DConfig.  Defaults to ``JEPA3DConfig()``.
    """

    def _get_default_jepa_config(self) -> JEPA3DConfig:
        return JEPA3DConfig()

    def build_architecture_and_adaptation_plan(
        self,
        config_plan: ConfigurationPlan,
        num_input_channels: int,
        num_output_channels: int,
    ) -> Tuple[nn.Module, AdaptationPlan]:
        """Build the JEPA3DModel and create the downstream adaptation plan.

        Args:
            config_plan: Configuration plan.
            num_input_channels: Number of input channels.
            num_output_channels: Not used.

        Returns:
            Tuple of (JEPA3DModel, AdaptationPlan).
        """
        model = JEPA3DModel(config=self.jepa_config)

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans(arch_class_name="ResEncL"),
            pretrain_plan=self.plan,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            pretrain_num_input_channels=self.jepa_config.in_channels,
            key_to_encoder="context_encoder.layer2",
            key_to_stem="context_encoder.stem",
            keys_to_in_proj=("context_encoder.stem.0",),
        )
        return model, adapt_plan


# ---------------------------------------------------------------------------
# Convenience variants for different batch sizes
# ---------------------------------------------------------------------------


class IJEPATrainer_BS4(IJEPATrainer):
    """I-JEPA trainer with batch size 4."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4


class IJEPATrainer_BS8(IJEPATrainer):
    """I-JEPA trainer with batch size 8."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_batch_size = 8


class JEPA3DTrainer_BS2(JEPA3DTrainer):
    """3D-JEPA trainer with batch size 2 (for memory-constrained GPUs)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_batch_size = 2


class JEPA3DTrainer_BS4(JEPA3DTrainer):
    """3D-JEPA trainer with batch size 4."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.total_batch_size = 4
