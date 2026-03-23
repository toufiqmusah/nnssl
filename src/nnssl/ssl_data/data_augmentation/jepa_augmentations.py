"""
JEPA-specific 3D augmentations for medical image self-supervised pretraining.

These augmentations are designed for the JEPA training paradigm where
consistent spatial structure is important (unlike contrastive methods that
apply independent strong augmentations to each view).

Augmentations included:
- JEPASpatialAugmentation3D: Rotation, scaling, flipping
- JEPAIntensityAugmentation3D: Brightness, contrast, gamma, Gaussian noise
- JEPAElasticDeformation3D: Elastic deformations for medical images
- JEPAAugmentationPipeline: Combined pipeline for JEPA training
"""

from typing import Optional, Tuple

import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import (
    BrightnessMultiplicativeTransform,
    ContrastAugmentationTransform,
    GammaTransform,
)
from batchgenerators.transforms.noise_transforms import (
    GaussianBlurTransform,
    GaussianNoiseTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.utility_transforms import NumpyToTensor


class JEPASpatialAugmentation3D(AbstractTransform):
    """Spatial augmentations for JEPA training.

    Applies random flipping and mild spatial transforms.  Augmentations are
    intentionally *milder* than in contrastive methods because JEPA requires
    the context and target to share meaningful spatial structure.

    Args:
        do_mirror: Whether to apply random axis mirroring.
        mirror_axes: Axes to mirror along (0-indexed spatial axes).
        do_spatial: Whether to apply mild spatial transforms (rotation/scale).
        rotation_range: Range of rotation angles in radians.
        scale_range: Range of scaling factors.
        data_key: Key in the data dict containing the image volume.
    """

    def __init__(
        self,
        do_mirror: bool = True,
        mirror_axes: Tuple[int, ...] = (0, 1, 2),
        do_spatial: bool = True,
        rotation_range: float = np.pi / 12,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        data_key: str = "data",
    ) -> None:
        self._transforms = []

        if do_mirror:
            self._transforms.append(MirrorTransform(axes=mirror_axes))

        if do_spatial:
            self._transforms.append(
                SpatialTransform(
                    patch_size=None,  # determined dynamically
                    do_elastic_deform=False,
                    do_rotation=True,
                    angle_x=(-rotation_range, rotation_range),
                    angle_y=(-rotation_range, rotation_range),
                    angle_z=(-rotation_range, rotation_range),
                    do_scale=True,
                    scale=(scale_range[0], scale_range[1]),
                    border_mode_data="constant",
                    border_cval_data=0,
                    order_data=3,
                    random_crop=False,
                    p_rot_per_sample=0.2,
                    p_scale_per_sample=0.2,
                    p_el_per_sample=0.0,
                    data_key=data_key,
                )
            )

        self._compose = Compose(self._transforms)

    def __call__(self, **data_dict) -> dict:
        return self._compose(**data_dict)


class JEPAIntensityAugmentation3D(AbstractTransform):
    """Intensity augmentations for JEPA training.

    Applies mild brightness, contrast, gamma, and Gaussian noise augmentations.
    These do not change spatial structure, so they are safe to apply without
    disrupting the block-masking scheme.

    Args:
        p_noise: Probability of applying Gaussian noise.
        p_blur: Probability of applying Gaussian blur.
        p_brightness: Probability of applying brightness augmentation.
        p_contrast: Probability of applying contrast augmentation.
        p_gamma: Probability of applying gamma augmentation.
        data_key: Key in the data dict containing the image volume.
    """

    def __init__(
        self,
        p_noise: float = 0.1,
        p_blur: float = 0.2,
        p_brightness: float = 0.15,
        p_contrast: float = 0.15,
        p_gamma: float = 0.1,
        data_key: str = "data",
    ) -> None:
        self._transforms = Compose(
            [
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_sample=p_noise,
                    data_key=data_key,
                ),
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.0),
                    different_sigma_per_channel=True,
                    p_per_sample=p_blur,
                    p_per_channel=0.5,
                    data_key=data_key,
                ),
                BrightnessMultiplicativeTransform(
                    multiplier_range=(0.75, 1.25),
                    p_per_sample=p_brightness,
                    data_key=data_key,
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.75, 1.25),
                    p_per_sample=p_contrast,
                    data_key=data_key,
                ),
                GammaTransform(
                    gamma_range=(0.7, 1.5),
                    invert_image=False,
                    per_channel=True,
                    retain_stats=True,
                    p_per_sample=p_gamma,
                    data_key=data_key,
                ),
            ]
        )

    def __call__(self, **data_dict) -> dict:
        return self._transforms(**data_dict)


class JEPAAugmentationPipeline(AbstractTransform):
    """Combined spatial + intensity augmentation pipeline for JEPA training.

    Chains spatial and intensity augmentations, then converts the numpy
    array to a float32 PyTorch tensor.

    Args:
        do_spatial_aug: Whether to include spatial augmentations.
        do_intensity_aug: Whether to include intensity augmentations.
        rotation_range: Maximum rotation angle (radians) for spatial aug.
        scale_range: Scaling factor range for spatial aug.
        p_noise: Probability for Gaussian noise augmentation.
        p_blur: Probability for Gaussian blur augmentation.
        p_brightness: Probability for brightness augmentation.
        p_contrast: Probability for contrast augmentation.
        p_gamma: Probability for gamma augmentation.
        data_key: Key for the image tensor in the data dict.
        to_tensor: Whether to convert to PyTorch tensor at the end.
    """

    def __init__(
        self,
        do_spatial_aug: bool = True,
        do_intensity_aug: bool = True,
        rotation_range: float = np.pi / 12,
        scale_range: Tuple[float, float] = (0.85, 1.15),
        p_noise: float = 0.1,
        p_blur: float = 0.2,
        p_brightness: float = 0.15,
        p_contrast: float = 0.15,
        p_gamma: float = 0.1,
        data_key: str = "data",
        to_tensor: bool = True,
    ) -> None:
        transforms = []

        if do_spatial_aug:
            transforms.append(
                JEPASpatialAugmentation3D(
                    do_mirror=True,
                    do_spatial=True,
                    rotation_range=rotation_range,
                    scale_range=scale_range,
                    data_key=data_key,
                )
            )

        if do_intensity_aug:
            transforms.append(
                JEPAIntensityAugmentation3D(
                    p_noise=p_noise,
                    p_blur=p_blur,
                    p_brightness=p_brightness,
                    p_contrast=p_contrast,
                    p_gamma=p_gamma,
                    data_key=data_key,
                )
            )

        if to_tensor:
            transforms.append(NumpyToTensor([data_key], "float"))

        self._pipeline = Compose(transforms) if transforms else None

    def __call__(self, **data_dict) -> dict:
        if self._pipeline is not None:
            return self._pipeline(**data_dict)
        return data_dict


class JEPAValidationTransform(AbstractTransform):
    """Minimal transform for JEPA validation — converts numpy to tensor only.

    No augmentations are applied during validation.  The block mask generator
    still runs (inside the trainer's ``validation_step``) so validation loss
    is comparable to training loss.

    Args:
        data_key: Key for the image tensor in the data dict.
    """

    def __init__(self, data_key: str = "data") -> None:
        self._to_tensor = NumpyToTensor([data_key], "float")

    def __call__(self, **data_dict) -> dict:
        return self._to_tensor(**data_dict)
