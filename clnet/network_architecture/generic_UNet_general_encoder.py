#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np

import torch
from torch import nn
import torch.nn.functional as functional

from clnet.network_architecture.initialization import InitWeights_He
from clnet.network_architecture.generic_UNet import ConvDropoutNormNonlin, StackedConvLayers, BasicResBlock


class Generic_UNet_General_Encoder(nn.Module):
    '''
    The downstream path (include bottleneck) of Generic_UNet, return skips of selected levels.
    '''
    MAX_NUM_FILTERS_3D = 320
    MAX_FILTERS_2D = 480

    def __init__(self, input_channels, base_num_features, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d, norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None, conv_kernel_sizes=None,
                 convolutional_pooling=False, convolutional_upsampling=False, max_num_features=None,
                 basic_block=ConvDropoutNormNonlin, if_full_network=False):
        super(Generic_UNet_General_Encoder, self).__init__()
        self.if_full_network = if_full_network
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
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
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op

        if conv_op == nn.Conv2d:
            pool_op = nn.MaxPool2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            pool_op = nn.MaxPool3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
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

        self.conv_blocks_context = []
        self.td = []

        output_features = base_num_features
        input_features = input_channels
        if isinstance(num_conv_per_stage, int):
            num_conv_per_stage = [num_conv_per_stage] * (num_pool + 1)
        for d in range(num_pool):
            # determine the first stride
            if d != 0 and self.convolutional_pooling:
                first_stride = pool_op_kernel_sizes[d - 1]
            else:
                first_stride = None

            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(StackedConvLayers(input_features, output_features, num_conv_per_stage[d],
                                                              self.conv_op, self.conv_kwargs, self.norm_op,
                                                              self.norm_op_kwargs, self.dropout_op,
                                                              self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                                              first_stride, basic_block=basic_block))
            if not self.convolutional_pooling:
                self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = int(np.round(output_features * feat_map_mul_on_downscale))
            output_features = min(output_features, self.max_num_features)

        # now the bottleneck.
        # determine the first stride
        if self.convolutional_pooling:
            first_stride = pool_op_kernel_sizes[-1]
        else:
            first_stride = None

        # the output of the last conv must match the number of features from the skip connection if we are not using
        # convolutional upsampling. If we use convolutional upsampling then the reduction in feature maps will be
        # done by the transposed conv
        if self.convolutional_upsampling:
            final_num_features = output_features
        else:
            final_num_features = self.conv_blocks_context[-1].output_channels

        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(nn.Sequential(
            StackedConvLayers(input_features, output_features, num_conv_per_stage[-1] - 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, first_stride, basic_block=basic_block),
            StackedConvLayers(output_features, final_num_features, 1, self.conv_op, self.conv_kwargs,
                              self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                              self.nonlin_kwargs, basic_block=basic_block)))

        # Here, we consider if we wish to have a "full" network, instead of just a encoding path.
        if if_full_network:
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
                    pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
                if conv_kernel_sizes is None:
                    conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
            else:
                raise ValueError("unknown convolution dimensionality, conv op: %s" % str(conv_op))

            self.conv_blocks_localization = []
            self.tu = []
            self.seg_outputs = []

            conv_blocks_context = self.conv_blocks_context
            nfeature_previous_block = conv_blocks_context[-1][-1].output_channels
            self.num_level = (len(conv_blocks_context) - 1) if num_pool > (len(conv_blocks_context) - 1) else num_pool

            # if we don't want to do dropout in the localization pathway then we set the dropout prob to zero here
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0
            # now lets build the localization pathway
            for u in range(num_pool):
                # The skip channel number
                nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels

                # The channel number of the decoding head
                nfeatures_out_decoder = nfeatures_from_skip
                nfeatures_in_decoder = nfeatures_out_decoder + nfeatures_from_skip

                # The channel number of the self.tu
                nfeatures_in_tu = nfeature_previous_block
                nfeatures_out_tu = nfeatures_out_decoder
                # Setup the self.tu using TransConv
                self.tu.append(transpconv(nfeatures_in_tu, nfeatures_out_tu, pool_op_kernel_sizes[-(u + 1)],
                                          pool_op_kernel_sizes[-(u + 1)], bias=False))

                self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[- (u + 1)]
                self.conv_kwargs['padding'] = self.conv_pad_sizes[- (u + 1)]

                self.conv_blocks_localization.append(nn.Sequential(
                    StackedConvLayers(nfeatures_in_decoder, nfeatures_out_decoder,
                                      num_conv_per_stage[- (u + 2)] - 1, self.conv_op, self.conv_kwargs, self.norm_op,
                                      self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs, self.nonlin,
                                      self.nonlin_kwargs, basic_block=basic_block),
                    StackedConvLayers(nfeatures_out_decoder, nfeatures_out_decoder, 1,
                                      self.conv_op, self.conv_kwargs,
                                      self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                                      self.nonlin, self.nonlin_kwargs, basic_block=basic_block)
                ))
                self.tu = nn.ModuleList(self.tu)
                self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
                nfeature_previous_block = nfeatures_out_decoder

        # register all modules properly
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)

    def forward(self, x, skip_feat_list=None):
        if skip_feat_list is None or len(skip_feat_list) == 0:
            skip_feat_list = range(len(self.conv_blocks_context))  # return all skip features by default

        skips = []  # encoder features, include bottleneck output (deepest feature)
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            if not self.convolutional_pooling:
                x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)
        skips.append(x)
        skips = [skips[i] for i in skip_feat_list]

        if self.if_full_network:
            full_network_skips = []
            skips_offset = 1  # ! last skip feat is used, set offset to 1
            for u in range(len(self.tu)):
                x = self.tu[u](x)
                f_skip = skips[-(u + 1 + skips_offset)]
                x = torch.cat((x, f_skip), dim=1)
                x = self.conv_blocks_localization[u](x)
                full_network_skips.insert(0, x)
            full_network_skips.append(skips[-1])
            skips = full_network_skips
        return skips
