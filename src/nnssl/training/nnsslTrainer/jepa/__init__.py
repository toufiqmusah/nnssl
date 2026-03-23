"""
JEPA (Joint-Embedding Predictive Architecture) module for nnssl.

This module provides I-JEPA and 3D-JEPA self-supervised pretraining methods
for 3D medical images, following the JEPA framework of Assran et al. (2023).

Public API
----------
Models:
    IJEPAModel          Full I-JEPA model (context encoder + target encoder + predictor)
    JEPA3DModel         3D-JEPA model with CNN-based volumetric encoder
    VisionTransformer3D 3D Vision Transformer backbone
    ContextEncoder3D    Context encoder (processes visible patches)
    JEPAPredictor       Predictor network

Loss:
    VICRegLoss          Variance-Invariance-Covariance Regularization loss

Data:
    BlockMaskGenerator3D   Generates 3D block masks at patch resolution
    JEPATransform          Data transform for I-JEPA
    JEPA3DTransform        Data transform for 3D-JEPA

Trainers:
    IJEPATrainer        I-JEPA trainer (ViT-based)
    JEPA3DTrainer       3D-JEPA trainer (CNN-based)
    IJEPATrainer_BS4    Batch-size 4 variant of IJEPATrainer
    IJEPATrainer_BS8    Batch-size 8 variant of IJEPATrainer
    JEPA3DTrainer_BS2   Batch-size 2 variant of JEPA3DTrainer
    JEPA3DTrainer_BS4   Batch-size 4 variant of JEPA3DTrainer

Configuration:
    IJEPAConfig         Complete I-JEPA configuration
    JEPA3DConfig        Complete 3D-JEPA configuration
    VICRegLossConfig    VICReg loss configuration
    EncoderConfig       Encoder configuration
    PredictorConfig     Predictor configuration
    MaskingConfig       Masking strategy configuration
"""

from nnssl.training.nnsslTrainer.jepa.jepa_config import (
    JEPA3DConfig,
    IJEPAConfig,
    EncoderConfig,
    MaskingConfig,
    PredictorConfig,
    VICRegLossConfig,
)
from nnssl.training.nnsslTrainer.jepa.jepa_data import (
    BlockMaskGenerator3D,
    JEPA3DTransform,
    JEPATransform,
    MultiScaleBlockMaskGenerator3D,
)
from nnssl.training.nnsslTrainer.jepa.jepa_loss import VICRegLoss
from nnssl.training.nnsslTrainer.jepa.jepa_models import (
    JEPA3DModel,
    IJEPAModel,
    JEPAPredictor,
    PatchEmbed3D,
    VisionTransformer3D,
)
from nnssl.training.nnsslTrainer.jepa.jepa_trainer import (
    JEPA3DTrainer,
    JEPA3DTrainer_BS2,
    JEPA3DTrainer_BS4,
    IJEPATrainer,
    IJEPATrainer_BS4,
    IJEPATrainer_BS8,
)

__all__ = [
    # Models
    "IJEPAModel",
    "JEPA3DModel",
    "VisionTransformer3D",
    "PatchEmbed3D",
    "JEPAPredictor",
    # Loss
    "VICRegLoss",
    # Data
    "BlockMaskGenerator3D",
    "MultiScaleBlockMaskGenerator3D",
    "JEPATransform",
    "JEPA3DTransform",
    # Trainers
    "IJEPATrainer",
    "JEPA3DTrainer",
    "IJEPATrainer_BS4",
    "IJEPATrainer_BS8",
    "JEPA3DTrainer_BS2",
    "JEPA3DTrainer_BS4",
    # Configuration
    "IJEPAConfig",
    "JEPA3DConfig",
    "VICRegLossConfig",
    "EncoderConfig",
    "PredictorConfig",
    "MaskingConfig",
]
