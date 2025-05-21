import numpy as np
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from typing import Union, Tuple, Callable
from batchgenerators.augmentations.resample_augmentations import augment_linear_downsampling_scipy
from batchgenerators.augmentations.color_augmentations import augment_gamma, augment_brightness_multiplicative, \
    augment_contrast
from batchgenerators.augmentations.noise_augmentations import augment_gaussian_blur, augment_gaussian_noise
from batchgenerators.transforms.color_transforms import augment_brightness_additive


class SelectiveChannelBrightnessTransform(AbstractTransform):
    def __init__(self, mu, sigma, per_channel=True, data_key="data", p_per_sample=1, p_per_channel=1, idx_aug=None):
        """
        Augments the brightness of data. Additive brightness is sampled from Gaussian distribution with mu and sigma
        :param mu: mean of the Gaussian distribution to sample the added brightness from
        :param sigma: standard deviation of the Gaussian distribution to sample the added brightness from
        :param per_channel: whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.mu = mu
        self.sigma = sigma
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_brightness_additive(
                        data_dict[self.data_key][b][self.idx_aug], self.mu, self.sigma, self.per_channel,
                        p_per_channel=self.p_per_channel)
                else:
                    data_dict[self.data_key][b] = augment_brightness_additive(data_dict[self.data_key][b], self.mu,
                                                                              self.sigma, self.per_channel,
                                                                              p_per_channel=self.p_per_channel)
        return data_dict


class SelectiveChannelSimulateLowResolutionTransform(AbstractTransform):
    """Downsamples each sample (linearly) by a random factor and upsamples to original resolution again
    (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel:

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:
    """

    def __init__(self, zoom_range=(0.5, 1), per_channel=False, p_per_channel=1,
                 channels=None, order_downsample=1, order_upsample=0, data_key="data", p_per_sample=1,
                 ignore_axes=None, idx_aug=None):
        self.order_upsample = order_upsample
        self.order_downsample = order_downsample
        self.channels = channels
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.zoom_range = zoom_range
        self.ignore_axes = ignore_axes
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_linear_downsampling_scipy(
                        data_dict[self.data_key][b][self.idx_aug],
                        zoom_range=self.zoom_range,
                        per_channel=self.per_channel,
                        p_per_channel=self.p_per_channel,
                        channels=self.channels,
                        order_downsample=self.order_downsample,
                        order_upsample=self.order_upsample,
                        ignore_axes=self.ignore_axes)
                else:
                    data_dict[self.data_key][b] = augment_linear_downsampling_scipy(data_dict[self.data_key][b],
                                                                                    zoom_range=self.zoom_range,
                                                                                    per_channel=self.per_channel,
                                                                                    p_per_channel=self.p_per_channel,
                                                                                    channels=self.channels,
                                                                                    order_downsample=self.order_downsample,
                                                                                    order_upsample=self.order_upsample,
                                                                                    ignore_axes=self.ignore_axes)
        return data_dict


class SelectiveChannelContrastAugmentationTransform(AbstractTransform):
    def __init__(self,
                 contrast_range: Union[Tuple[float, float], Callable[[], float]] = (0.75, 1.25),
                 preserve_range: bool = True,
                 per_channel: bool = True,
                 data_key: str = "data",
                 p_per_sample: float = 1,
                 p_per_channel: float = 1,
                 idx_aug=None):
        """
        Augments the contrast of data
        :param contrast_range:
            (float, float): range from which to sample a random contrast that is applied to the data. If
                            one value is smaller and one is larger than 1, half of the contrast modifiers will be >1
                            and the other half <1 (in the inverval that was specified)
            callable      : must be contrast_range() -> float
        :param preserve_range: if True then the intensity values after contrast augmentation will be cropped to min and
        max values of the data before augmentation.
        :param per_channel: whether to use the same contrast modifier for all color channels or a separate one for each
        channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.contrast_range = contrast_range
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_contrast(
                        data_dict[self.data_key][b][self.idx_aug],
                        contrast_range=self.contrast_range,
                        preserve_range=self.preserve_range,
                        per_channel=self.per_channel,
                        p_per_channel=self.p_per_channel)
                else:
                    data_dict[self.data_key][b] = augment_contrast(data_dict[self.data_key][b],
                                                                   contrast_range=self.contrast_range,
                                                                   preserve_range=self.preserve_range,
                                                                   per_channel=self.per_channel,
                                                                   p_per_channel=self.p_per_channel)
        return data_dict


class SelectiveChannelBrightnessMultiplicativeTransform(AbstractTransform):
    def __init__(self, multiplier_range=(0.5, 2), per_channel=True, data_key="data", p_per_sample=1, idx_aug=None):
        """
        Augments the brightness of data. Multiplicative brightness is sampled from multiplier_range
        :param multiplier_range: range to uniformly sample the brightness modifier from
        :param per_channel:  whether to use the same brightness modifier for all color channels or a separate one for
        each channel
        :param data_key:
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.multiplier_range = multiplier_range
        self.per_channel = per_channel
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_brightness_multiplicative(
                        data_dict[self.data_key][b][self.idx_aug],
                        self.multiplier_range,
                        self.per_channel)
                else:
                    data_dict[self.data_key][b] = augment_brightness_multiplicative(data_dict[self.data_key][b],
                                                                                    self.multiplier_range,
                                                                                    self.per_channel)
        return data_dict


class SelectiveChannelGaussianBlurTransform(AbstractTransform):
    def __init__(self, blur_sigma: Tuple[float, float] = (1, 5), different_sigma_per_channel: bool = True,
                 different_sigma_per_axis: bool = False, p_isotropic: float = 0, p_per_channel: float = 1,
                 p_per_sample: float = 1, data_key: str = "data", idx_aug=None):
        """

        :param blur_sigma:
        :param data_key:
        :param different_sigma_per_axis: if True, anisotropic kernels are possible
        :param p_isotropic: only applies if different_sigma_per_axis=True, p_isotropic is the proportion of isotropic
        kernels, the rest gets random sigma per axis
        :param different_sigma_per_channel: whether to sample a sigma for each channel or all channels at once
        :param p_per_channel: probability of applying gaussian blur for each channel. Default = 1 (all channels are
        blurred with prob 1)
        """
        self.p_per_sample = p_per_sample
        self.different_sigma_per_channel = different_sigma_per_channel
        self.p_per_channel = p_per_channel
        self.data_key = data_key
        self.blur_sigma = blur_sigma
        self.different_sigma_per_axis = different_sigma_per_axis
        self.p_isotropic = p_isotropic
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_gaussian_blur(
                        data_dict[self.data_key][b][self.idx_aug], self.blur_sigma,
                        self.different_sigma_per_channel,
                        self.p_per_channel,
                        different_sigma_per_axis=self.different_sigma_per_axis,
                        p_isotropic=self.p_isotropic)
                else:
                    data_dict[self.data_key][b] = augment_gaussian_blur(data_dict[self.data_key][b], self.blur_sigma,
                                                                        self.different_sigma_per_channel,
                                                                        self.p_per_channel,
                                                                        different_sigma_per_axis=self.different_sigma_per_axis,
                                                                        p_isotropic=self.p_isotropic)
        return data_dict


class SelectiveChannelGaussianNoiseTransform(AbstractTransform):
    def __init__(self, noise_variance=(0, 0.1), p_per_sample=1, p_per_channel: float = 1,
                 per_channel: bool = False, data_key="data", idx_aug=None):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_gaussian_noise(
                        data_dict[self.data_key][b][self.idx_aug], self.noise_variance, self.p_per_channel,
                        self.per_channel)
                else:
                    data_dict[self.data_key][b] = augment_gaussian_noise(data_dict[self.data_key][b],
                                                                         self.noise_variance,
                                                                         self.p_per_channel, self.per_channel)
        return data_dict


class SelectiveChannelGammaTransform(AbstractTransform):
    def __init__(self, gamma_range=(0.5, 2), invert_image=False, per_channel=False, data_key="data",
                 retain_stats: Union[bool, Callable[[], bool]] = False, p_per_sample=1, idx_aug=None):
        """
        Augments by changing 'gamma' of the image (same as gamma correction in photos or computer monitors

        :param gamma_range: range to sample gamma from. If one value is smaller than 1 and the other one is
        larger then half the samples will have gamma <1 and the other >1 (in the inverval that was specified).
        Tuple of float. If one value is < 1 and the other > 1 then half the images will be augmented with gamma values
        smaller than 1 and the other half with > 1
        :param invert_image: whether to invert the image before applying gamma augmentation
        :param per_channel:
        :param data_key:
        :param retain_stats: Gamma transformation will alter the mean and std of the data in the patch. If retain_stats=True,
        the data will be transformed to match the mean and standard deviation before gamma augmentation. retain_stats
        can also be callable (signature retain_stats() -> bool)
        :param p_per_sample:
        """
        self.p_per_sample = p_per_sample
        self.retain_stats = retain_stats
        self.per_channel = per_channel
        self.data_key = data_key
        self.gamma_range = gamma_range
        self.invert_image = invert_image
        self.idx_aug = idx_aug

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                if self.idx_aug is not None and len(self.idx_aug) > 0:
                    data_dict[self.data_key][b][self.idx_aug] = augment_gamma(data_dict[self.data_key][b][self.idx_aug],
                                                                              self.gamma_range,
                                                                              self.invert_image,
                                                                              per_channel=self.per_channel,
                                                                              retain_stats=self.retain_stats)
                else:
                    data_dict[self.data_key][b] = augment_gamma(data_dict[self.data_key][b], self.gamma_range,
                                                                self.invert_image,
                                                                per_channel=self.per_channel,
                                                                retain_stats=self.retain_stats)
        return data_dict


class MoveAuxData(AbstractTransform):
    """
    Move Aux Data to 'seg' or 'data'
    """

    def __init__(self, key_origin="data", key_target="seg", remove_from_origin=True):
        self.remove_from_origin = remove_from_origin
        self.key_target = key_target
        self.key_origin = key_origin
        self.num_of_image_input = 1

    def __call__(self, **data_dict):
        origin = data_dict.get(self.key_origin)
        target = data_dict.get(self.key_target)
        aux_data = origin[:, self.num_of_image_input:]
        target = np.concatenate((target, aux_data), 1)
        data_dict[self.key_target] = target

        if self.remove_from_origin:
            origin = origin[:, :self.num_of_image_input]
            data_dict[self.key_origin] = origin
        return data_dict


class RemoveKeyTransform(AbstractTransform):
    def __init__(self, key_to_remove):
        self.key_to_remove = key_to_remove

    def __call__(self, **data_dict):
        _ = data_dict.pop(self.key_to_remove, None)
        return data_dict


class MaskTransform(AbstractTransform):
    def __init__(self, dct_for_where_it_was_used, mask_idx_in_seg=1, set_outside_to=0, data_key="data", seg_key="seg"):
        """
        data[mask < 0] = 0
        Sets everything outside the mask to 0. CAREFUL! outside is defined as < 0, not =0 (in the Mask)!!!

        :param dct_for_where_it_was_used:
        :param mask_idx_in_seg:
        :param set_outside_to:
        :param data_key:
        :param seg_key:
        """
        self.dct_for_where_it_was_used = dct_for_where_it_was_used
        self.seg_key = seg_key
        self.data_key = data_key
        self.set_outside_to = set_outside_to
        self.mask_idx_in_seg = mask_idx_in_seg

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        if seg is None or seg.shape[1] < self.mask_idx_in_seg:
            raise Warning("mask not found, seg may be missing or seg[:, mask_idx_in_seg] may not exist")
        data = data_dict.get(self.data_key)
        for b in range(data.shape[0]):
            mask = seg[b, self.mask_idx_in_seg]
            for c in range(data.shape[1]):
                if self.dct_for_where_it_was_used[c]:
                    data[b, c][mask < 0] = self.set_outside_to
        data_dict[self.data_key] = data
        return data_dict


def convert_3d_to_2d_generator(data_dict):
    shp = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_data'] = shp
    shp = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1] * shp[2], shp[3], shp[4]))
    data_dict['orig_shape_seg'] = shp
    return data_dict


def convert_2d_to_3d_generator(data_dict):
    shp = data_dict['orig_shape_data']
    current_shape = data_dict['data'].shape
    data_dict['data'] = data_dict['data'].reshape((shp[0], shp[1], shp[2], current_shape[-2], current_shape[-1]))
    shp = data_dict['orig_shape_seg']
    current_shape_seg = data_dict['seg'].shape
    data_dict['seg'] = data_dict['seg'].reshape((shp[0], shp[1], shp[2], current_shape_seg[-2], current_shape_seg[-1]))
    return data_dict


class Convert3DTo2DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_3d_to_2d_generator(data_dict)


class Convert2DTo3DTransform(AbstractTransform):
    def __init__(self):
        pass

    def __call__(self, **data_dict):
        return convert_2d_to_3d_generator(data_dict)


class ConvertSegmentationToRegionsTransform(AbstractTransform):
    def __init__(self, regions: dict, seg_key: str = "seg", output_key: str = "seg", seg_channel: int = 0):
        """
        regions are tuple of tuples where each inner tuple holds the class indices that are merged into one region, example:
        regions= ((1, 2), (2, )) will result in 2 regions: one covering the region of labels 1&2 and the other just 2
        :param regions:
        :param seg_key:
        :param output_key:
        """
        self.seg_channel = seg_channel
        self.output_key = output_key
        self.seg_key = seg_key
        self.regions = regions

    def __call__(self, **data_dict):
        seg = data_dict.get(self.seg_key)
        num_regions = len(self.regions)
        if seg is not None:
            seg_shp = seg.shape
            output_shape = list(seg_shp)
            output_shape[1] = num_regions
            region_output = np.zeros(output_shape, dtype=seg.dtype)
            for b in range(seg_shp[0]):
                for r, k in enumerate(self.regions.keys()):
                    for l in self.regions[k]:
                        region_output[b, r][seg[b, self.seg_channel] == l] = 1
            data_dict[self.output_key] = region_output
        return data_dict
