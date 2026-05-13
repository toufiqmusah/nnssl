"""
Brain-JEPA Trainer for nnssl.

Inherits from :class:`BaseEvaMAETrainer` to reuse the Primus-specific optimizer
schedule (warmup → PolyLR), grad clipping, and checkpoint logic.  Overrides
architecture construction, loss, data transforms, and train/validation steps
for JEPA semantics.

The key differences from MAE training are:
  1. The model's ``forward()`` returns ``(loss, info_dict)`` — no external loss.
  2. Masks (context + target blocks) are generated in the data pipeline.
  3. After each optimizer step, the target encoder receives an EMA update with
     a cosine-annealed decay schedule.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch
from torch import autocast, nn
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP

from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import save_json

from nnssl.adaptation_planning.adaptation_plan import AdaptationPlan, ArchitecturePlans
from nnssl.architectures.brain_jepa_module import BrainJEPA
from nnssl.experiment_planning.experiment_planners.plan import Plan
from nnssl.ssl_data.configure_basic_dummyDA import configure_rotation_dummyDA_mirroring_and_inital_patch_size
from nnssl.ssl_data.data_augmentation.transforms_for_dummy_2d import Convert2DTo3DTransform, Convert3DTo2DTransform
from nnssl.ssl_data.dataloading.jepa_transform import JEPAMaskTransform
from nnssl.training.nnsslTrainer.masked_image_modeling.BaseEvaMAETrainer import BaseEvaMAETrainer
from nnssl.utilities.helpers import dummy_context


class BaseBrainJEPATrainer(BaseEvaMAETrainer):
    """JEPA pre-training with Primus encoders inside nnssl.

    Inherits optimizer schedule, checkpoint save/load, and DDP setup from
    :class:`BaseEvaMAETrainer`.
    """

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device,
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)

        # Override patch size for JEPA (same default as MAE Primus-M)
        self.config_plan.patch_size = (160, 160, 160)
        self.vit_patch_size = (8, 8, 8)
        self.embed_dim = 864          # Primus-M defaults
        self.encoder_eva_depth = 16
        self.encoder_eva_numheads = 12

        # JEPA-specific hyperparameters
        self.stage_indices = None     # None = auto thirds of eva_depth
        self.pred_dim = 384
        self.pred_depth = 4
        self.pred_heads = 6
        self.ema_decay_start = 0.996
        self.ema_decay_end = 1.0
        self.num_target_blocks = 4
        self.stage_loss_weights = None  # None → [0.5, 0.75, 1.0]

        # Step counter for EMA cosine schedule
        self._global_step = 0
        self._total_steps = None  # set in initialize()

        # Will be created in get_dataloaders
        self._jepa_mask_transform = None

    # ------------------------------------------------------------------ #
    #  Architecture
    # ------------------------------------------------------------------ #

    def build_architecture_and_adaptation_plan(
        self, config_plan, num_input_channels, num_output_channels
    ) -> tuple[nn.Module, AdaptationPlan]:
        network = BrainJEPA(
            primus_kwargs=dict(
                input_channels=1,
                embed_dim=self.embed_dim,
                patch_embed_size=self.vit_patch_size,
                eva_depth=self.encoder_eva_depth,
                eva_numheads=self.encoder_eva_numheads,
                input_shape=tuple(self.config_plan.patch_size),
                drop_path_rate=self.drop_path_rate,
                init_values=self.init_value,
                scale_attn_inner=True,
            ),
            stage_indices=self.stage_indices,
            pred_dim=self.pred_dim,
            pred_depth=self.pred_depth,
            pred_heads=self.pred_heads,
            ema_decay_start=self.ema_decay_start,
            ema_decay_end=self.ema_decay_end,
            stage_loss_weights=self.stage_loss_weights,
        )

        adapt_plan = AdaptationPlan(
            architecture_plans=ArchitecturePlans("PrimusM"),
            pretrain_plan=self.plan,
            pretrain_num_input_channels=1,
            recommended_downstream_patchsize=self.recommended_downstream_patchsize,
            # These point into the context encoder's Primus weights
            key_to_encoder="context_encoder.eva",
            key_to_stem="context_encoder.down_projection",
            keys_to_in_proj=("context_encoder.down_projection.proj",),
            key_to_lpe="context_encoder.eva.pos_embed",
        )
        save_json(adapt_plan.serialize(), self.adaptation_json_plan)
        return network, adapt_plan

    # ------------------------------------------------------------------ #
    #  Loss (internal to model)
    # ------------------------------------------------------------------ #

    def build_loss(self):
        """Loss is computed inside BrainJEPA.forward() — no external loss."""
        return None

    # ------------------------------------------------------------------ #
    #  Initialization
    # ------------------------------------------------------------------ #

    def initialize(self):
        super().initialize()
        # Compute total steps for EMA cosine schedule
        self._total_steps = self.num_epochs * self.num_iterations_per_epoch

    # ------------------------------------------------------------------ #
    #  Data pipeline
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_training_transforms(
        patch_size,
        rotation_for_DA,
        mirror_axes,
        do_dummy_2d_data_aug,
        order_resampling_data=3,
        order_resampling_seg=1,
        border_val_seg=-1,
        use_mask_for_norm=None,
        _jepa_mask_transform=None,
    ) -> AbstractTransform:
        """Build training transforms, with JEPA mask appended at the end.

        The ``_jepa_mask_transform`` kwarg is injected by :meth:`get_dataloaders`
        via the override pattern — see below.
        """
        tr_transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                alpha=(0, 0),
                sigma=(0, 0),
                do_rotation=True,
                angle_x=rotation_for_DA["x"],
                angle_y=rotation_for_DA["y"],
                angle_z=rotation_for_DA["z"],
                p_rot_per_axis=1,
                do_scale=True,
                scale=(0.7, 1.4),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                border_mode_seg="constant",
                border_cval_seg=border_val_seg,
                order_seg=order_resampling_seg,
                random_crop=False,
                p_el_per_sample=0,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
                independent_scale_for_each_axis=False,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(NumpyToTensor(["data"], "float"))
        tr_transforms.append(NumpyToTensor(["seg"], "long"))

        # Append JEPA mask sampling as the final transform
        if _jepa_mask_transform is not None:
            tr_transforms.append(_jepa_mask_transform)

        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_dataloaders(self):
        """Override to pass the JEPA mask transform into the static method."""
        # Compute patch grid for the mask sampler
        patch_grid = tuple(
            s // p for s, p in zip(self.config_plan.patch_size, self.vit_patch_size)
        )
        self._jepa_mask_transform = JEPAMaskTransform(
            patch_grid=patch_grid,
            num_target_blocks=self.num_target_blocks,
        )

        # Now we override get_training_transforms for this instance to inject
        # the mask transform. We do this by monkey-patching the static method
        # call site in the parent's get_dataloaders.
        patch_size = self.config_plan.patch_size
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = configure_rotation_dummyDA_mirroring_and_inital_patch_size(patch_size)

        from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
        from nnssl.ssl_data.limited_len_wrapper import LimitedLenWrapper
        from nnssl.utilities.default_n_proc_DA import get_allowed_n_proc_DA

        if do_dummy_2d_data_aug:
            self.print_to_log_file("Using dummy 2D data augmentation")

        # Build transforms WITH the JEPA mask transform injected
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            mirror_axes,
            do_dummy_2d_data_aug,
            order_resampling_data=3,
            order_resampling_seg=1,
            use_mask_for_norm=getattr(self.config_plan, 'use_mask_for_norm', None),
            _jepa_mask_transform=self._jepa_mask_transform,
        )

        # Validation transforms — add mask transform too (for val loss)
        val_transforms = self.get_validation_transforms(
            _jepa_mask_transform=self._jepa_mask_transform,
        )

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

    @staticmethod
    def get_validation_transforms(_jepa_mask_transform=None) -> AbstractTransform:
        """Validation transforms with optional JEPA mask sampling."""
        val_transforms = []
        val_transforms.append(NumpyToTensor(["data"], "float"))
        val_transforms.append(NumpyToTensor(["seg"], "long"))
        if _jepa_mask_transform is not None:
            val_transforms.append(_jepa_mask_transform)
        val_transforms = Compose(val_transforms)
        return val_transforms

    # ------------------------------------------------------------------ #
    #  Validation epoch start — keep masking active
    # ------------------------------------------------------------------ #

    def on_validation_epoch_start(self):
        """Don't switch to eval mode — masking must stay active."""
        pass

    # ------------------------------------------------------------------ #
    #  Train step
    # ------------------------------------------------------------------ #

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        data = data.to(self.device, non_blocking=True)

        ctx_mask = torch.from_numpy(batch["context_mask"]).bool().to(
            self.device, non_blocking=True
        )
        tgt_masks = torch.from_numpy(batch["target_masks"]).bool().to(
            self.device, non_blocking=True
        )

        self.optimizer.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            loss, info = self.network(data, ctx_mask, tgt_masks)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad],
                self.grad_clip,
            )
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.network.parameters() if p.requires_grad],
                self.grad_clip,
            )
            self.optimizer.step()

        # EMA update with cosine-annealed tau (must use unwrapped module for DDP)
        net = self.network.module if self.is_ddp else self.network
        if isinstance(net, OptimizedModule):
            net = net._orig_mod
        net.update_target_encoder(
            step=self._global_step, total_steps=self._total_steps
        )
        self._global_step += 1

        return {
            "loss": loss.detach().cpu().numpy(),
            **{k: np.float32(v) for k, v in info.items()},
        }

    # ------------------------------------------------------------------ #
    #  Validation step
    # ------------------------------------------------------------------ #

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        data = data.to(self.device, non_blocking=True)

        ctx_mask = torch.from_numpy(batch["context_mask"]).bool().to(
            self.device, non_blocking=True
        )
        tgt_masks = torch.from_numpy(batch["target_masks"]).bool().to(
            self.device, non_blocking=True
        )

        with autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            loss, info = self.network(data, ctx_mask, tgt_masks)

        return {
            "loss": loss.detach().cpu().numpy(),
            **{k: np.float32(v) for k, v in info.items()},
        }

    # ------------------------------------------------------------------ #
    #  Checkpoint loading (inherits from BaseEvaMAETrainer)
    #  The full BrainJEPA state dict — including target_encoder — is saved
    #  and restored correctly.  _global_step is re-derived from current_epoch.
    # ------------------------------------------------------------------ #

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        super().load_checkpoint(filename_or_checkpoint)
        # Restore global step from the loaded epoch
        self._global_step = self.current_epoch * self.num_iterations_per_epoch


# ====================================================================== #
#  Experiment Variant Subclasses (§6.7)
# ====================================================================== #


class BaseBrainJEPATrainer_PrimusM_160ps(BaseBrainJEPATrainer):
    """Standard Primus-M JEPA, 160³ patches, BS=8."""

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (160, 160, 160)
        self.total_batch_size = 8


class BaseBrainJEPATrainer_PrimusS_96ps(BaseBrainJEPATrainer):
    """Primus-S, 96³ — faster iteration / ablations."""

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.embed_dim = 396
        self.encoder_eva_depth = 12
        self.encoder_eva_numheads = 6
        self.pred_dim = 198  # ~half of embed_dim
        self.config_plan.patch_size = (96, 96, 96)
        self.total_batch_size = 8


class BaseBrainJEPATrainer_test(BaseBrainJEPATrainer):
    """Smoke-test variant: 2 epochs, BS=2, 96³."""

    def __init__(
        self,
        plan: Plan,
        configuration_name: str,
        fold: int,
        pretrain_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plan, configuration_name, fold, pretrain_json, device)
        self.config_plan.patch_size = (96, 96, 96)
        self.embed_dim = 396
        self.encoder_eva_depth = 12
        self.encoder_eva_numheads = 6
        self.pred_dim = 198
        self.total_batch_size = 2
        self.num_epochs = 2
