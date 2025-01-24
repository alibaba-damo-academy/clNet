#   Author @Dazhou Guo
#   Data: 03.01.2023

import numpy as np
from typing import Union, Tuple

import torch
from torch import nn
import torch.nn.functional
from torch.amp import autocast

from clnet.configuration import default_base_num_feature, default_pool, default_conv, default_num_conv_per_stage, \
    default_alpha_ema_encoder, default_alpha_ema_decoder, default_gpu_ram_constraint
from clnet.utilities.nd_softmax import softmax_helper
from clnet.utilities.to_torch import to_cuda, maybe_to_torch
from clnet.network_architecture.initialization import InitWeights_He, InitZero
from clnet.network_architecture.neural_network import SegmentationNetwork
from clnet.network_architecture.generic_UNet import ConvDropoutNormNonlin, ResConvDropoutNormNonlin
from clnet.network_architecture.generic_UNet_supp import Generic_UNet_Supporting_Organ
from clnet.network_architecture.generic_UNet_general_encoder import Generic_UNet_General_Encoder
from clnet.network_architecture.generic_UNet_decoder_ensemble import Generic_UNet_Decoder_Ensemble
from clnet.network_architecture.custom_modules.pruning_modules import perform_unstructured_lottery_ticket_pruning, perform_masking_ema_model

from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import *


class Generic_UNet_Continual_Base(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    # BASE_NUM_FEATURES_3D = 30
    BASE_NUM_FEATURES_3D = 32
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000

    def __init__(self, task_dict, input_channels, base_num_features, num_pool=None, feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
                 nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False, final_nonlin=softmax_helper,
                 weightInitializer=InitWeights_He(1e-2), upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):
        """
        combine encoder and decoder, can add more decoders
        """
        super(Generic_UNet_Continual_Base, self).__init__()
        self.conv_op = conv_op
        self.decoder_dict = nn.ModuleDict({})
        self.supporting_dict = nn.ModuleDict({})
        self.ema_dict = nn.ModuleDict({})
        self.task_dict = task_dict
        self.pool_op_kernel_size_general_encoder = None
        self.alpha_encoder = default_alpha_ema_encoder
        self.alpha_decoder = default_alpha_ema_decoder

        # if self.task_dict["batch_size"]
        # Default setting
        image_channels = input_channels
        num_conv_per_stage = default_num_conv_per_stage
        conv_kernel_sizes = default_conv
        pool_op_kernel_sizes = default_pool

        for task in task_dict:
            train_dict = task_dict[task]
            pth_plans_file = task_dict[task]["plans_file"]
            stage = task_dict[task]["stage"]
            if "encoder_architecture_setup" in train_dict and train_dict["encoder_architecture_setup"] is not None and train_dict["type"] == "GeneralEncoder":
                num_conv_per_stage, conv_kernel_sizes, pool_op_kernel_sizes = self._load_plans(pth_plans_file, stage)
                if "num_conv_per_stage" in train_dict["encoder_architecture_setup"] and \
                        train_dict["encoder_architecture_setup"]["num_conv_per_stage"] is not None:
                    num_conv_per_stage = train_dict["encoder_architecture_setup"]["num_conv_per_stage"]
                if "conv_kernel" in train_dict["encoder_architecture_setup"] and \
                        train_dict["encoder_architecture_setup"]["conv_kernel"] is not None:
                    conv_kernel_sizes = train_dict["encoder_architecture_setup"]["conv_kernel"]
                if "pool_kernel" in train_dict["encoder_architecture_setup"] and \
                        train_dict["encoder_architecture_setup"]["pool_kernel"] is not None:
                    pool_op_kernel_sizes = train_dict["encoder_architecture_setup"]["pool_kernel"]
                if "base_num_features" in train_dict["encoder_architecture_setup"] and \
                        train_dict["encoder_architecture_setup"]["base_num_features"] is not None:
                    base_num_features = train_dict["encoder_architecture_setup"]["base_num_features"]

            # Setup the General-Encoder
            if train_dict["type"] == "GeneralEncoder":
                self.pool_op_kernel_size_general_encoder = pool_op_kernel_sizes
                if_full_network = self.task_dict[task]["full_network"]
                # Setup general encoder architecture
                self.encoder = Generic_UNet_General_Encoder(
                    image_channels, base_num_features, num_pool, num_conv_per_stage=num_conv_per_stage,
                    feat_map_mul_on_downscale=feat_map_mul_on_downscale, conv_op=conv_op, norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                    nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, weightInitializer=weightInitializer,
                    pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                    convolutional_pooling=convolutional_pooling, convolutional_upsampling=convolutional_upsampling,
                    max_num_features=max_num_features, basic_block=basic_block, if_full_network=if_full_network)
                # use the "enable_ema" bool defined in cfg json to indicate if or not enable EMA update
                if train_dict["encoder_architecture_setup"]["enable_ema"]:
                    # Setup the EMA encoder
                    self.ema_dict["general_encoder"] = Generic_UNet_General_Encoder(
                        image_channels, base_num_features, num_pool, num_conv_per_stage=num_conv_per_stage,
                        feat_map_mul_on_downscale=feat_map_mul_on_downscale, conv_op=conv_op, norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                        nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, weightInitializer=InitZero,
                        pool_op_kernel_sizes=pool_op_kernel_sizes, conv_kernel_sizes=conv_kernel_sizes,
                        convolutional_pooling=convolutional_pooling, convolutional_upsampling=convolutional_upsampling,
                        max_num_features=max_num_features, basic_block=basic_block, if_full_network=if_full_network)
                    # Initialize the key encoder with the same weights as the query encoder
                    self.ema_dict["general_encoder"].load_state_dict(self.encoder.state_dict())

            self.max_decoder_level = num_pool
            # Setup the decoding head architecture
            for decoder in train_dict["decoders"]:
                if decoder in task_dict[task]["decoder_architecture_setup"] and task_dict[task]["decoder_architecture_setup"][decoder] is not None:
                    decoder_architecture_setup_from_train_json = task_dict[task]["decoder_architecture_setup"][decoder]
                    current_decoder_base_num_feature = int(decoder_architecture_setup_from_train_json["base_num_feature"])
                    num_conv_per_stage = decoder_architecture_setup_from_train_json["num_conv_per_stage"]
                    if decoder_architecture_setup_from_train_json["conv_kernel"] is not None:
                        conv_kernel_sizes = decoder_architecture_setup_from_train_json["conv_kernel"]
                    current_enable_ema = decoder_architecture_setup_from_train_json["enable_ema"]

                    current_num_classes = train_dict["decoders"][decoder]
                    if isinstance(current_num_classes, list):
                        current_num_classes = len(current_num_classes) + 1
                    elif isinstance(current_num_classes, int):
                        current_num_classes = 2
                    else:
                        raise RuntimeError("Class" + decoder + "is wrongly defined")
                    if decoder not in self.decoder_dict:
                        # Here, we force the "pool_op_kernel_size" be the same, s.t., matching the skip features dimensions
                        if self.pool_op_kernel_size_general_encoder != pool_op_kernel_sizes:
                            pool_op_kernel_sizes = self.pool_op_kernel_size_general_encoder
                        current_decoder = Generic_UNet_Decoder_Ensemble(
                            current_decoder_base_num_feature, self.encoder, self.max_decoder_level, current_num_classes,
                            num_pool, num_conv_per_stage=num_conv_per_stage, conv_op=conv_op, norm_op=norm_op,
                            norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, deep_supervision=deep_supervision,
                            dropout_in_localization=dropout_in_localization, final_nonlin=final_nonlin,
                            weight_initializer=weightInitializer, pool_op_kernel_sizes=pool_op_kernel_sizes,
                            conv_kernel_sizes=conv_kernel_sizes, upscale_logits=upscale_logits,
                            convolutional_upsampling=convolutional_upsampling, max_num_features=max_num_features,
                            basic_block=basic_block, seg_output_use_bias=seg_output_use_bias)

                        self.decoder_dict[decoder] = current_decoder
                        if current_enable_ema:
                            self.ema_dict[decoder] = Generic_UNet_Decoder_Ensemble(
                                current_decoder_base_num_feature, self.encoder, self.max_decoder_level, current_num_classes,
                                num_pool, num_conv_per_stage=num_conv_per_stage, conv_op=conv_op, norm_op=norm_op,
                                norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
                                nonlin=nonlin, nonlin_kwargs=nonlin_kwargs, deep_supervision=deep_supervision,
                                dropout_in_localization=dropout_in_localization, final_nonlin=final_nonlin,
                                weight_initializer=InitZero, pool_op_kernel_sizes=pool_op_kernel_sizes,
                                conv_kernel_sizes=conv_kernel_sizes, upscale_logits=upscale_logits,
                                convolutional_upsampling=convolutional_upsampling, max_num_features=max_num_features,
                                basic_block=basic_block, seg_output_use_bias=seg_output_use_bias)
                            self.ema_dict[decoder].load_state_dict(current_decoder.state_dict())
            # Note: currently we do not support EMA updates in Feature-Level-Supporting
            if "supporting" in train_dict and train_dict["supporting"] is not None and len(train_dict["supporting"]) > 0:
                for decoder in train_dict["supporting"]:
                    if decoder in self.supporting_dict:
                        # I will ALLOW the changes in supporting organs.
                        # We can always use different supporting organ for help.
                        print("Duplicated supporting head -- '%s' in Task '%s'" % (decoder, task))
                        if self.supporting_dict[decoder] != train_dict["supporting"]:
                            print("The supporting of", decoder, "has changed FROM", self.supporting_dict[decoder],
                                  "TO", train_dict["supporting"])
                    if decoder in self.decoder_dict:
                        current_supporting = Generic_UNet_Supporting_Organ(
                            self.decoder_dict, train_dict["supporting"], decoder, num_pool, conv_op=conv_op,
                            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs, dropout_op=dropout_op,
                            dropout_op_kwargs=dropout_op_kwargs, nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
                            weight_initializer=weightInitializer)
                    else:
                        raise RuntimeError("Decoder for '%s' is NOT found!" % decoder)
                    self.supporting_dict[decoder] = current_supporting
            self.task_dict[task] = train_dict
        self.do_ds = True
        if weightInitializer is not None:
            self.apply(weightInitializer)

    def update_ema_encoder(self):
        param_encoder = {name: param for name, param in self.encoder.named_parameters()}
        for ema_name, ema_param in self.ema_dict["general_encoder"].named_parameters():
            # print(ema_name, ema_name in param_encoder)
            if ema_name in param_encoder:
                ema_param.data.mul_(self.alpha_encoder).add_(param_encoder[ema_name].data, alpha=1 - self.alpha_encoder)

    def update_ema_decoder(self, decoder):
        param_decoders = {name: param for name, param in self.decoder_dict[decoder].named_parameters()}
        for ema_name, ema_param in self.ema_dict[decoder].named_parameters():
            if ema_name in param_decoders:
                ema_param.data.mul_(self.alpha_decoder).add_(param_decoders[ema_name].data, alpha=1 - self.alpha_decoder)

    def apply_pruning(self, head_to_train, initial_state_dict, global_sparsity):
        decoder_to_prune = self.decoder_dict[head_to_train]
        perform_unstructured_lottery_ticket_pruning(decoder_to_prune, initial_state_dict, global_sparsity)

    def apply_pruning_mask_to_ema(self, head_to_train):
        if head_to_train == "all":
            for head in self.train_dict["decoders"]:
                if head in self.ema_dict:
                    perform_masking_ema_model(self.decoder_dict[head], self.ema_dict[head])
        else:
            if head_to_train in self.ema_dict and head_to_train in self.decoder_dict:
                perform_masking_ema_model(self.decoder_dict[head_to_train], self.ema_dict[head_to_train])

    def _load_plans(self, pth_plans_file, stage):
        def convert_keys_to_int(d: dict):
            new_dict = {}
            for k, v in d.items():
                try:
                    new_key = int(k)
                except ValueError:
                    new_key = k
                if type(v) == dict:
                    v = convert_keys_to_int(v)
                new_dict[new_key] = v
            return new_dict

        plans = convert_keys_to_int(load_json(pth_plans_file))
        num_conv_per_stage = plans["conv_per_stage"]
        conv_kernel_sizes = plans["plans_per_stage"][stage]["conv_kernel_sizes"]
        pool_op_kernel_sizes = plans["plans_per_stage"][stage]["pool_op_kernel_sizes"]
        return num_conv_per_stage, conv_kernel_sizes, pool_op_kernel_sizes

    def set_do_ds(self, do_ds):
        self.do_ds = do_ds
        for dec_name, dec_module in self.decoder_dict.items():
            self.decoder_dict[dec_name].do_ds = do_ds

    def set_to_eval_mode(self, train_dict, decoder_or_support, head_to_train):
        # If the task is not "GeneralEncoder", then we set "self.encoder" to "eval()"
        # If "require_grad=False", then the "eval()" only affect "Dropout", "BatchNorm"
        encoder_or_decoder = train_dict["type"]
        if encoder_or_decoder != "GeneralEncoder" or decoder_or_support == "supporting":
            self.encoder.eval()
        if head_to_train == "all":
            if decoder_or_support == "decoders":
                # If the current task is decoder, then we set the entire "supporting_dict" to "eval()"
                for head_name, su_module in self.supporting_dict.named_modules():
                    su_module.eval()
            else:
                # If the current task is supporting,
                # We let all decoder heads in "supporting_dict" remain the previous state
                # We let supporting heads in the current task's "supporting" remain the previous state
                for head_name, de_module in self.decoder_dict.named_modules():
                    if head_name not in train_dict["supporting"]:
                        de_module.eval()
                for head_name, de_module in self.supporting_dict.named_modules():
                    if head_name not in train_dict["supporting"]:
                        de_module.eval()
        else:
            if decoder_or_support == "decoders":
                # If the current task is decoder, then we first set the entire "supporting_dict" to "eval()"
                for head_name, su_module in self.supporting_dict.items():
                    su_module.eval()
                # Then, we only let "head_to_train" in "decoder_dict" remain the previous state
                for head_name, de_module in self.decoder_dict.named_modules():
                    if head_name != head_to_train:
                        de_module.eval()
            else:
                # If the current task is supporting,
                # We only let the "head_to_train" remain the previous state
                for head_name, de_module in self.decoder_dict.named_modules():
                    if head_name != head_to_train:
                        de_module.eval()
                # Then, we only let "head_to_train" in "supporting_dict" remain the previous state
                for head_name, su_module in self.supporting_dict.named_modules():
                    if head_name != head_to_train:
                        su_module.eval()

    def forward(self, x, train_dict, decoder_or_support, head_to_train):
        if train_dict["type"] == "GeneralEncoder":
            skips = self.encoder(x, skip_feat_list=None)
        else:
            with torch.no_grad():
                skips = self.encoder(x, skip_feat_list=None)
        seg_feats = [skips.pop()]
        ret_outputs = {}
        flag_missing = None
        if head_to_train == "all":
            if decoder_or_support == "decoders":
                for head in train_dict["decoders"]:
                    if head in self.decoder_dict:
                        ret_outputs[head] = self.decoder_dict[head](skips, seg_feats=seg_feats)
                    else:
                        flag_missing = head
            else:
                for head in train_dict["supporting"]:
                    if head in self.supporting_dict:
                        ret_outputs[head] = self.supporting_dict[head](self.decoder_dict, skips, seg_feats=seg_feats)
                    else:
                        flag_missing = head
        else:
            if decoder_or_support == "decoders":
                if head_to_train in self.decoder_dict:
                    ret_outputs[head_to_train] = self.decoder_dict[head_to_train](skips, seg_feats=seg_feats)
                else:
                    flag_missing = head_to_train
            else:
                if head_to_train in self.supporting_dict:
                    ret_outputs[head_to_train] = self.supporting_dict[head_to_train](self.decoder_dict, skips, seg_feats=seg_feats)
                else:
                    flag_missing = head_to_train
        if flag_missing is not None:
            raise RuntimeError("The decoder head %s is not available." % flag_missing)
        return ret_outputs

    def predict_3D_ensemble(self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
                            do_mirroring: bool, mirror_axes: Tuple[int, ...] = (0, 1, 2),
                            use_sliding_window: bool = False, step_size: float = 0.5,
                            patch_size: Tuple[int, ...] = None, regions_class_order: Tuple[int, ...] = None,
                            use_gaussian: bool = False, pad_border_mode: str = "constant",
                            pad_kwargs: dict = None, all_in_gpu: bool = False,
                            verbose: bool = True, mixed_precision: bool = True, ram_in_byte: int = 0) -> Tuple[np.ndarray, np.ndarray]:

        torch.cuda.empty_cache()
        assert step_size <= 1, "step_size must be smaller than 1. Otherwise there will be a gap between consecutive predictions"

        if verbose:
            print("debug: mirroring", do_mirroring, "mirror_axes", mirror_axes)

        if pad_kwargs is None:
            pad_kwargs = {'constant_values': 0}

        # A very long time ago the mirror axes were (2, 3, 4) for a 3d network. This is just to intercept any old
        # code that uses this convention
        if len(mirror_axes):
            if self.conv_op == nn.Conv2d:
                if max(mirror_axes) > 1:
                    raise ValueError("mirror axes. duh")
            if self.conv_op == nn.Conv3d:
                if max(mirror_axes) > 2:
                    raise ValueError("mirror axes. duh")

        if self.training:
            print('WARNING! Network is in train mode during inference. This may be intended, or not...')

        assert len(x.shape) == 4, "data must have shape (c,x,y,z)"

        if torch.cuda.is_available():
            current_device = "cuda"
        else:
            current_device = "cpu"
        # ###############################################################
        # The first class is always the "background".
        self.num_classes = 1
        if head_to_train == "all":
            for decoder in train_dict["decoders"]:
                if isinstance(train_dict["decoders"][decoder], list):
                    self.num_classes += len(train_dict["decoders"][decoder])
                else:
                    self.num_classes += 1
        else:
            if head_to_train in train_dict["decoders"]:
                if isinstance(train_dict["decoders"][head_to_train], list):
                    self.num_classes += len(train_dict["decoders"][head_to_train])
                else:
                    self.num_classes += 1
        # we check if the input data is too big, e.g., >12GB RAM. If it is, we do not run the inference on GPU
        precision_byte = 4
        if x.dtype == np.float16:
            precision_byte = 2
        elif x.dtype == np.float64:
            precision_byte = 8
        if 2 * np.prod(x.shape) * (self.num_classes + 2) * precision_byte > default_gpu_ram_constraint * (1024 ** 3) - ram_in_byte:
            all_in_gpu = False
            print("Input image is too big for GPU inference. Running on CPU instead.")
        # ###############################################################
        with autocast(device_type=current_device):
            with torch.no_grad():
                if self.conv_op == nn.Conv3d:
                    if use_sliding_window:
                        res = self._internal_predict_3D_3Dconv_tiled_ensemble(
                            x, train_dict, decoder_or_support, head_to_train, step_size, do_mirroring,
                            mirror_axes, patch_size, regions_class_order, use_gaussian, pad_border_mode,
                            pad_kwargs=pad_kwargs, all_in_gpu=all_in_gpu, verbose=verbose)
                    else:
                        res = self._internal_predict_3D_3Dconv_ensemble(
                            x, train_dict, decoder_or_support, head_to_train, patch_size, do_mirroring, mirror_axes,
                            regions_class_order, pad_border_mode, pad_kwargs=pad_kwargs, verbose=verbose)
                elif self.conv_op == nn.Conv2d:
                    patch_size = patch_size[-2:]
                    if use_sliding_window:
                        res = self._internal_predict_3D_2Dconv_tiled_ensemble(
                            x, train_dict, decoder_or_support, head_to_train, patch_size, do_mirroring,
                            mirror_axes, step_size, regions_class_order, use_gaussian, pad_border_mode,
                            pad_kwargs, all_in_gpu, False)
                    else:
                        res = self._internal_predict_3D_2Dconv_ensemble(
                            x, train_dict, decoder_or_support, head_to_train, patch_size, do_mirroring,
                            mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, all_in_gpu, False)
                else:
                    raise RuntimeError("Invalid conv op, cannot determine what dimensionality (2d/3d) the network is")
        return res

    def _internal_predict_3D_3Dconv_tiled_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str, step_size: float,
            do_mirroring: bool, mirror_axes: tuple, patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
            pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        if verbose:
            print("step_size:", step_size)
        if verbose:
            print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y, z

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_3d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_3d)]):
                if verbose:
                    print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_3d = gaussian_importance_map
                self._patch_size_for_gaussian_3d = patch_size
            else:
                if verbose:
                    print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_3d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            # predict on cpu if cuda not available
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)
        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose:
                print("initializing result array (on GPU)")
            aggregated_results = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())

            if verbose:
                print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose:
                print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())

        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_3d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]
                for z in steps[2]:
                    lb_z = z
                    ub_z = z + patch_size[2]

                    predicted_patch = \
                        self._internal_maybe_mirror_and_pred_3D_ensemble(
                            data[None, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z], train_dict,
                            decoder_or_support, head_to_train, mirror_axes, do_mirroring, gaussian_importance_map)[0]

                    if all_in_gpu:
                        predicted_patch = predicted_patch.half()
                    else:
                        predicted_patch = predicted_patch.cpu().numpy()

                    aggregated_results[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += predicted_patch
                    aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple([slice(0, aggregated_results.shape[i])
                        for i in range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose:
                print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose:
            print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_2D_2Dconv_tiled_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str, step_size: float,
            do_mirroring: bool, mirror_axes: tuple, patch_size: tuple, regions_class_order: tuple, use_gaussian: bool,
            pad_border_mode: str, pad_kwargs: dict, all_in_gpu: bool, verbose: bool) -> Tuple[np.ndarray, np.ndarray]:
        # better safe than sorry
        assert len(x.shape) == 3, "x must be (c, x, y)"

        if verbose: print("step_size:", step_size)
        if verbose: print("do mirror:", do_mirroring)

        assert patch_size is not None, "patch_size cannot be None for tiled prediction"

        # for sliding window inference the image must at least be as large as the patch size. It does not matter
        # whether the shape is divisible by 2**num_pool as long as the patch size is
        data, slicer = pad_nd_image(x, patch_size, pad_border_mode, pad_kwargs, True, None)
        data_shape = data.shape  # still c, x, y

        # compute the steps for sliding window
        steps = self._compute_steps_for_sliding_window(patch_size, data_shape[1:], step_size)
        num_tiles = len(steps[0]) * len(steps[1])

        if verbose:
            print("data shape:", data_shape)
            print("patch size:", patch_size)
            print("steps (x, y, and z):", steps)
            print("number of tiles:", num_tiles)

        # we only need to compute that once. It can take a while to compute this due to the large sigma in
        # gaussian_filter
        if use_gaussian and num_tiles > 1:
            if self._gaussian_2d is None or not all(
                    [i == j for i, j in zip(patch_size, self._patch_size_for_gaussian_2d)]):
                if verbose: print('computing Gaussian')
                gaussian_importance_map = self._get_gaussian(patch_size, sigma_scale=1. / 8)

                self._gaussian_2d = gaussian_importance_map
                self._patch_size_for_gaussian_2d = patch_size
            else:
                if verbose: print("using precomputed Gaussian")
                gaussian_importance_map = self._gaussian_2d

            gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
            if torch.cuda.is_available():
                gaussian_importance_map = gaussian_importance_map.cuda(self.get_device(), non_blocking=True)

        else:
            gaussian_importance_map = None

        if all_in_gpu:
            # If we run the inference in GPU only (meaning all tensors are allocated on the GPU, this reduces
            # CPU-GPU communication but required more GPU memory) we need to preallocate a few things on GPU

            if use_gaussian and num_tiles > 1:
                # half precision for the outputs should be good enough. If the outputs here are half, the
                # gaussian_importance_map should be as well
                gaussian_importance_map = gaussian_importance_map.half()

                # make sure we did not round anything to 0
                gaussian_importance_map[gaussian_importance_map == 0] = gaussian_importance_map[
                    gaussian_importance_map != 0].min()

                add_for_nb_of_preds = gaussian_importance_map
            else:
                add_for_nb_of_preds = torch.ones(patch_size, device=self.get_device())

            if verbose:
                print("initializing result array (on GPU)")
            aggregated_results = \
                torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())

            if verbose:
                print("moving data to GPU")
            data = torch.from_numpy(data).cuda(self.get_device(), non_blocking=True)

            if verbose:
                print("initializing result_numsamples (on GPU)")
            aggregated_nb_of_predictions = \
                torch.zeros([self.num_classes] + list(data.shape[1:]), dtype=torch.half, device=self.get_device())
        else:
            if use_gaussian and num_tiles > 1:
                add_for_nb_of_preds = self._gaussian_2d
            else:
                add_for_nb_of_preds = np.ones(patch_size, dtype=np.float32)
            aggregated_results = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)
            aggregated_nb_of_predictions = np.zeros([self.num_classes] + list(data.shape[1:]), dtype=np.float32)

        for x in steps[0]:
            lb_x = x
            ub_x = x + patch_size[0]
            for y in steps[1]:
                lb_y = y
                ub_y = y + patch_size[1]

                predicted_patch = self._internal_maybe_mirror_and_pred_2D_ensemble(
                    data[None, :, lb_x:ub_x, lb_y:ub_y], train_dict, decoder_or_support, head_to_train,
                    mirror_axes, do_mirroring, gaussian_importance_map)[0]

                if all_in_gpu:
                    predicted_patch = predicted_patch.half()
                else:
                    predicted_patch = predicted_patch.cpu().numpy()

                aggregated_results[:, lb_x:ub_x, lb_y:ub_y] += predicted_patch
                aggregated_nb_of_predictions[:, lb_x:ub_x, lb_y:ub_y] += add_for_nb_of_preds

        # we reverse the padding here (remeber that we padded the input to be at least as large as the patch size
        slicer = tuple(
            [slice(0, aggregated_results.shape[i]) for i in
             range(len(aggregated_results.shape) - (len(slicer) - 1))] + slicer[1:])
        aggregated_results = aggregated_results[slicer]
        aggregated_nb_of_predictions = aggregated_nb_of_predictions[slicer]

        # computing the class_probabilities by dividing the aggregated result with result_numsamples
        class_probabilities = aggregated_results / aggregated_nb_of_predictions

        if regions_class_order is None:
            predicted_segmentation = class_probabilities.argmax(0)
        else:
            if all_in_gpu:
                class_probabilities_here = class_probabilities.detach().cpu().numpy()
            else:
                class_probabilities_here = class_probabilities
            predicted_segmentation = np.zeros(class_probabilities_here.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[class_probabilities_here[i] > 0.5] = c

        if all_in_gpu:
            if verbose: print("copying results to CPU")

            if regions_class_order is None:
                predicted_segmentation = predicted_segmentation.detach().cpu().numpy()

            class_probabilities = class_probabilities.detach().cpu().numpy()

        if verbose: print("prediction done")
        return predicted_segmentation, class_probabilities

    def _internal_predict_3D_2Dconv_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
            min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple = (0, 1),
            regions_class_order: tuple = None, pad_border_mode: str = "constant", pad_kwargs: dict = None,
            all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:

        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_ensemble(
                x[:, s], train_dict, decoder_or_support, head_to_train, min_size, do_mirroring, mirror_axes,
                regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_3Dconv_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
            min_size: Tuple[int, ...], do_mirroring: bool, mirror_axes: tuple = (0, 1, 2),
            regions_class_order: tuple = None, pad_border_mode: str = "constant", pad_kwargs: dict = None,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 4, "x must be (c, x, y, z)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_3D_3Dconv'
        if verbose: print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_3D_ensemble(
            data[None], train_dict, decoder_or_support, head_to_train, mirror_axes, do_mirroring, None)[0]

        slicer = tuple([slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) - (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_predict_3D_2Dconv_tiled_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
            patch_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple = (0, 1), step_size: float = 0.5,
            regions_class_order: tuple = None, use_gaussian: bool = False, pad_border_mode: str = "edge",
            pad_kwargs: dict = None, all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError

        assert len(x.shape) == 4, "data must be c, x, y, z"

        predicted_segmentation = []
        softmax_pred = []

        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_tiled_ensemble(
                x[:, s], train_dict, decoder_or_support, head_to_train, step_size, do_mirroring, mirror_axes,
                patch_size, regions_class_order, use_gaussian, pad_border_mode, pad_kwargs, all_in_gpu, verbose)

            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])

        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))

        return predicted_segmentation, softmax_pred

    def _internal_predict_3D_2Dconv_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
            min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple = (0, 1),
            regions_class_order: tuple = None, pad_border_mode: str = "constant", pad_kwargs: dict = None,
            all_in_gpu: bool = False, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if all_in_gpu:
            raise NotImplementedError
        assert len(x.shape) == 4, "data must be c, x, y, z"
        predicted_segmentation = []
        softmax_pred = []
        for s in range(x.shape[1]):
            pred_seg, softmax_pres = self._internal_predict_2D_2Dconv_ensemble(
                x[:, s], train_dict, decoder_or_support, head_to_train, min_size, do_mirroring,
                mirror_axes, regions_class_order, pad_border_mode, pad_kwargs, verbose)
            predicted_segmentation.append(pred_seg[None])
            softmax_pred.append(softmax_pres[None])
        predicted_segmentation = np.vstack(predicted_segmentation)
        softmax_pred = np.vstack(softmax_pred).transpose((1, 0, 2, 3))
        return predicted_segmentation, softmax_pred

    @staticmethod
    def _concatenate_multi_decoding_head_pred(pred_decoders):
        pred = None
        if isinstance(pred_decoders, dict):
            for decoder in pred_decoders:
                if pred is None:
                    pred = pred_decoders[decoder]
                else:
                    pred_classes = pred_decoders[decoder][:, 1:]
                    if len(pred.shape) != len(pred_classes.shape):
                        pred_classes = torch.unsqueeze(pred_classes, 1)
                    pred = torch.cat((pred, pred_classes), 1)
        return pred

    def _internal_predict_2D_2Dconv_ensemble(
            self, x: np.ndarray, train_dict: dict, decoder_or_support: str, head_to_train: str,
            min_size: Tuple[int, int], do_mirroring: bool, mirror_axes: tuple = (0, 1, 2),
            regions_class_order: tuple = None, pad_border_mode: str = "constant", pad_kwargs: dict = None,
            verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        This one does fully convolutional inference. No sliding window
        """
        assert len(x.shape) == 3, "x must be (c, x, y)"

        assert self.input_shape_must_be_divisible_by is not None, 'input_shape_must_be_divisible_by must be set to ' \
                                                                  'run _internal_predict_2D_2Dconv'
        if verbose:
            print("do mirror:", do_mirroring)

        data, slicer = pad_nd_image(x, min_size, pad_border_mode, pad_kwargs, True,
                                    self.input_shape_must_be_divisible_by)

        predicted_probabilities = self._internal_maybe_mirror_and_pred_2D_ensemble(
            data[None], train_dict, decoder_or_support, head_to_train, mirror_axes, do_mirroring, None)[0]

        slicer = tuple(
            [slice(0, predicted_probabilities.shape[i]) for i in range(len(predicted_probabilities.shape) -
                                                                       (len(slicer) - 1))] + slicer[1:])
        predicted_probabilities = predicted_probabilities[slicer]

        if regions_class_order is None:
            predicted_segmentation = predicted_probabilities.argmax(0)
            predicted_segmentation = predicted_segmentation.detach().cpu().numpy()
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
        else:
            predicted_probabilities = predicted_probabilities.detach().cpu().numpy()
            predicted_segmentation = np.zeros(predicted_probabilities.shape[1:], dtype=np.float32)
            for i, c in enumerate(regions_class_order):
                predicted_segmentation[predicted_probabilities[i] > 0.5] = c

        return predicted_segmentation, predicted_probabilities

    def _internal_maybe_mirror_and_pred_3D_ensemble(self, x: Union[np.ndarray, torch.tensor], train_dict: dict,
                                                    decoder_or_support: str, head_to_train: str, mirror_axes: tuple,
                                                    do_mirroring: bool = True, mult: np.ndarray or torch.tensor = None
                                                    ) -> torch.tensor:
        assert len(x.shape) == 5, 'x must be (b, c, x, y, z)'

        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        x = maybe_to_torch(x)
        result_torch = torch.zeros([1, self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 8
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                # pred_decoders = self(x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(x, train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * pred

            if m == 1 and (2 in mirror_axes):
                # tmp_x = torch.flip(x, (4,))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4,)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (4,))

            if m == 2 and (1 in mirror_axes):
                # tmp_x = torch.flip(x, (3,))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3,)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 3 and (2 in mirror_axes) and (1 in mirror_axes):
                # tmp_x = torch.flip(x, (4, 3))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (4, 3))

            if m == 4 and (0 in mirror_axes):
                # tmp_x = torch.flip(x, (2,))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (2,)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 5 and (0 in mirror_axes) and (2 in mirror_axes):
                # tmp_x = torch.flip(x, (4, 2))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 2)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (4, 2))

            if m == 6 and (0 in mirror_axes) and (1 in mirror_axes):
                # tmp_x = torch.flip(x, (3, 2))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

            if m == 7 and (0 in mirror_axes) and (1 in mirror_axes) and (2 in mirror_axes):
                # tmp_x = torch.flip(x, (4, 3, 2))
                # pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                # pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                pred = self.inference_apply_nonlin(self(torch.flip(x, (4, 3, 2)), train_dict, decoder_or_support, head_to_train)[head_to_train])
                result_torch += 1 / num_results * torch.flip(pred, (4, 3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch

    def _internal_maybe_mirror_and_pred_2D_ensemble(
            self, x: Union[np.ndarray, torch.tensor], train_dict: dict, decoder_or_support: str, head_to_train: str,
            mirror_axes: tuple, do_mirroring: bool = True, mult: np.ndarray or torch.tensor = None) -> torch.tensor:
        # if cuda available:
        #   everything in here takes place on the GPU. If x and mult are not yet on GPU this will be taken care of here
        #   we now return a cuda tensor! Not numpy array!

        assert len(x.shape) == 4, 'x must be (b, c, x, y)'

        x = maybe_to_torch(x)
        result_torch = torch.zeros([x.shape[0], self.num_classes] + list(x.shape[2:]), dtype=torch.float)

        if torch.cuda.is_available():
            x = to_cuda(x, gpu_id=self.get_device())
            result_torch = result_torch.cuda(self.get_device(), non_blocking=True)

        if mult is not None:
            mult = maybe_to_torch(mult)
            if torch.cuda.is_available():
                mult = to_cuda(mult, gpu_id=self.get_device())

        if do_mirroring:
            mirror_idx = 4
            num_results = 2 ** len(mirror_axes)
        else:
            mirror_idx = 1
            num_results = 1

        for m in range(mirror_idx):
            if m == 0:
                pred_decoders = self(x, train_dict, decoder_or_support, head_to_train)
                pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                # pred = self.inference_apply_nonlin(self(x))
                result_torch += 1 / num_results * pred

            if m == 1 and (1 in mirror_axes):
                tmp_x = torch.flip(x, (3,))
                pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (3,))))
                result_torch += 1 / num_results * torch.flip(pred, (3,))

            if m == 2 and (0 in mirror_axes):
                tmp_x = torch.flip(x, (2,))
                pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (2,))))
                result_torch += 1 / num_results * torch.flip(pred, (2,))

            if m == 3 and (0 in mirror_axes) and (1 in mirror_axes):
                tmp_x = torch.flip(x, (3, 2))
                pred_decoders = self(tmp_x, train_dict, decoder_or_support, head_to_train)
                pred = self.inference_apply_nonlin(self._concatenate_multi_decoding_head_pred(pred_decoders))
                # pred = self.inference_apply_nonlin(self(torch.flip(x, (3, 2))))
                result_torch += 1 / num_results * torch.flip(pred, (3, 2))

        if mult is not None:
            result_torch[:, :] *= mult

        return result_torch
