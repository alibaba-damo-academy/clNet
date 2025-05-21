from torch import nn
import torch
import numpy as np

from clnet.configuration import default_max_num_features, default_pool, default_conv
from clnet.utilities.nd_softmax import softmax_helper
from clnet.network_architecture.initialization import InitWeights_He
from clnet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, Upsample, ResAdd
from clnet.network_architecture.center_crop import center_crop_feature_pairs


class Generic_UNet_Decoder_Ensemble(nn.Module):
    """
    The full upstream path (exclude bottleneck) of Generic_UNet, input skips from encoder.
    """
    MAX_NUM_FILTERS_3D = default_max_num_features
    MAX_FILTERS_2D = 480

    def __init__(self, decoder_base_num_feature, encoder, num_level, num_classes, num_pool, num_conv_per_stage=2,
                 conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None, dropout_op=nn.Dropout2d,
                 dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True,
                 dropout_in_localization=False, final_nonlin=softmax_helper,
                 weight_initializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None, upscale_logits=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin, seg_output_use_bias=False):
        """
        encoder: need encoder backbone (feature extractor) for the feature dimension at each level
        num_level: decoder structure params, the level number add to backbone,
                   clNet decoder-like structure, cannot exceed the depth of encoder
        """
        super(Generic_UNet_Decoder_Ensemble, self).__init__()
        decoder_base_num_feature = int(max(1, decoder_base_num_feature))
        self.decoder_base_num_feature = np.clip(decoder_base_num_feature * np.array([1, 2, 4, 8, 16, 32]), decoder_base_num_feature, self.MAX_NUM_FILTERS_3D)
        self.convolutional_upsampling = convolutional_upsampling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weight_initializer = weight_initializer
        self.num_pool = num_pool
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self.deep_supervision = deep_supervision
        self.do_ds = deep_supervision
        self.dim_mismatch = True

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = default_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = default_conv
        else:
            raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_localization = []
        # self.skip_conv = []
        # self.res_add = []
        self.tu = []
        self.seg_outputs = []
        # conv_kwargs_1x1 = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': True}
        conv_blocks_context = encoder.conv_blocks_context
        nfeature_previous_block = conv_blocks_context[-1][-1].output_channels
        self.num_level = (len(conv_blocks_context) - 1) if num_level > (len(conv_blocks_context) - 1) else num_level

        # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
        old_dropout_p = self.dropout_op_kwargs['p']
        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = 0.0
        # get num_con_per_stage
        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = [num_conv_per_stage] * (num_pool + 1)
        # now lets build the localization pathway
        for u in range(num_pool):
            # The skip channel number
            nfeatures_from_skip = conv_blocks_context[-(2 + u)].output_channels

            # The channel number of the decoding head
            nfeatures_out_decoder = self.decoder_base_num_feature[-(2 + u)]

            nfeatures_in_decoder = nfeatures_out_decoder + nfeatures_from_skip

            # The channel number of the self.tu
            nfeatures_in_tu = nfeature_previous_block
            nfeatures_out_tu = nfeatures_out_decoder
            # Setup the self.tu using TransConv
            self.tu.append(transpconv(nfeatures_in_tu, nfeatures_out_tu, pool_op_kernel_sizes[-(u + 1)], pool_op_kernel_sizes[-(u + 1)], bias=False))
            # self.skip_conv.append(
            #     StackedConvLayers(nfeatures_from_skip, nfeatures_from_skip, 1, self.conv_op, conv_kwargs_1x1, None, None,
            #                       self.dropout_op, self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
            # )
            # self.res_add.append(
            #     ResAdd(nfeatures_in_decoder, nfeatures_out_decoder, self.conv_op, conv_kwargs_1x1, self.nonlin, self.nonlin_kwargs)
            # )
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]

            if num_conv_per_stage[-(u + 2)] > 1:
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(nfeatures_in_decoder, nfeatures_out_decoder, num_conv_per_stage[-(u + 2)] - 1,
                                      self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op,
                                      self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                    StackedConvLayers(nfeatures_out_decoder, nfeatures_out_decoder, 1, self.conv_op, self.conv_kwargs, self.norm_op, self.norm_op_kwargs,
                                      self.dropout_op, self.dropout_op_kwargs, None, None, basic_block=basic_block)
                ))
            else:
                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(nfeatures_in_decoder, nfeatures_out_decoder, num_conv_per_stage[-(u + 2)] - 1, self.conv_op,
                                      self.conv_kwargs, self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                      self.nonlin, self.nonlin_kwargs, basic_block=basic_block),
                ))

            nfeature_previous_block = nfeatures_out_decoder

        for ds in range(len(self.conv_blocks_localization)):
            self.seg_outputs.append(conv_op(self.conv_blocks_localization[ds][-1].output_channels, num_classes, 1, 1, 0, 1, 1, seg_output_use_bias))

        self.upscale_logits_ops = []
        cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(num_pool - self.num_level, num_pool - 1):
            if self.upscale_logits:
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.tu = nn.ModuleList(self.tu)
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        # self.skip_conv = nn.ModuleList(self.skip_conv)
        # self.res_add = nn.ModuleList(self.res_add)

        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(self.upscale_logits_ops)
        if self.weight_initializer is not None:
            self.apply(self.weight_initializer)

    def forward(self, skips, x):
        """
        expect skips only contains corresponding skip features for decoder layers in this module
        """
        seg_outputs = []
        for u in range(self.num_pool):
            x = self.tu[u](x)
            # f_skip = self.skip_conv[u](skips[-(u + 1)])
            f_skip = skips[-(u + 1)]
            # center crop the possible dimension mismatch caused by TransposeConv/UpSample and Conv.
            x, f_skip = center_crop_feature_pairs(x, f_skip)

            x_cat = torch.cat((x, f_skip), 1)
            x = self.conv_blocks_localization[u](x_cat)
            # x = self.res_add[u](x, x_cat)
            seg_outputs.append(self.final_nonlin(self.seg_outputs[u](x)))

        if self.deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
