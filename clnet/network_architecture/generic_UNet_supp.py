#   Author @Dazhou Guo
#   Data: 03.01.2023


import torch
from torch import nn
import torch.nn.functional
import torch.nn.functional as functional
from clnet.network_architecture.initialization import InitWeights_He, InitZero
from clnet.network_architecture.generic_UNet import ConvDropoutNormNonlin, SEConvDropoutNormNonlin, ConvDropoutNormNonlinSE


class Generic_UNet_Supporting_Organ(nn.Module):
    def __init__(self, decoder_dict, train_supporting_dict, decoder, num_pool, conv_op=nn.Conv3d, conv_kwargs=None, norm_op=nn.InstanceNorm3d,
                 norm_op_kwargs=None, dropout_op=nn.Dropout3d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 weight_initializer=InitWeights_He(1e-2)):
        super(Generic_UNet_Supporting_Organ, self).__init__()
        self.weightInitializer = weight_initializer
        self.support_dict = nn.ModuleDict({})
        self.decoder = decoder
        self.dim_mismatch = True
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        nfeature_total_supporting_channel = 0
        for supporting_organ in train_supporting_dict[decoder]:
            if supporting_organ in decoder_dict:
                current_supporting_subnet = []
                for u in range(num_pool):
                    # nfeatures_supporting_organ = decoder_dict[supporting_organ].conv_blocks_localization[u][-1].output_channels
                    # nfeatures_to_support_organ = decoder_dict[decoder].conv_blocks_localization[u][-1].output_channels
                    # current_supporting_subnet.append(
                    #     ConvDropoutNormNonlin(nfeatures_supporting_organ, nfeatures_to_support_organ, conv_op, conv_kwargs,
                    #                           norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))

                    nfeatures_supporting_organ = decoder_dict[supporting_organ].seg_outputs[u].out_channels
                    if u == 0:
                        nfeature_total_supporting_channel += nfeatures_supporting_organ
                    current_supporting_subnet.append(
                        ConvDropoutNormNonlin(nfeatures_supporting_organ, nfeatures_supporting_organ, conv_op, conv_kwargs,
                                              norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
                self.support_dict[supporting_organ] = nn.ModuleList(current_supporting_subnet)
            else:
                txt_msg = "Supporting organ '" + supporting_organ + "' is NOT found!"
                raise RuntimeError(txt_msg)
        to_support = []
        for u in range(num_pool):
            # nfeatures_to_support_organ = decoder_dict[decoder].conv_blocks_localization[u][-1].output_channels
            # to_support.append(
            #     ConvDropoutNormNonlin((len(train_supporting_dict[decoder]) + 1) * nfeatures_to_support_organ, nfeatures_to_support_organ,
            #                           conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))
            nfeatures_to_support_organ = decoder_dict[decoder].conv_blocks_localization[u][-1].output_channels
            to_support.append(
                ConvDropoutNormNonlin(nfeature_total_supporting_channel + nfeatures_to_support_organ, nfeatures_to_support_organ,
                                      conv_op, conv_kwargs, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs))

        self.to_support = nn.ModuleList(to_support)
        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, decoder_dict, skips, seg_feats):
        decoder_to_train = decoder_dict[self.decoder]
        seg_outputs = []
        skips_offset = 0
        if seg_feats is None or len(seg_feats) == 0:
            x = skips[-1]
            skips_offset = 1  # ! last skip feat is used, set offset to 1
        else:
            x = seg_feats[-1]
        support_feats = {}
        # Get the supporting features from the supporting decoders
        # The decoders that require supporting is fine-tuned sequentially (pre-determined order).
        for supporting_organ in self.support_dict:
            # Note: for the supporting decoders, it only forwards, no updates,
            # I.e., no grads are back-propagated.
            support_x = x.clone()
            current_support_feats = []
            for u in range(len(decoder_to_train.tu)):
                support_x = decoder_dict[supporting_organ].tu[u](support_x)
                f_skip = skips[-(u + 1 + skips_offset)]
                support_x_dim = support_x.size()[2:]
                f_skip_dim = f_skip.size()[2:]
                if support_x_dim != f_skip_dim:
                    support_x = functional.interpolate(support_x, size=f_skip_dim, mode='trilinear', align_corners=True)
                    if self.dim_mismatch:
                        print("#######################################################################################")
                        print("##     DIMENSION 'skip' vs. 'SUPPORTING feature' MISMATCH -- Please double check     ##")
                        print("##  We interpolate the 'decoding feature' feature to match 'skip' feature dimension! ##")
                        print("#######################################################################################")
                        self.dim_mismatch = False
                support_x = torch.cat((support_x, f_skip), dim=1)
                support_x = decoder_dict[supporting_organ].conv_blocks_localization[u](support_x)
                current_support_feats.append(decoder_dict[supporting_organ].final_nonlin(decoder_dict[supporting_organ].seg_outputs[u](support_x)))
            support_feats[supporting_organ] = current_support_feats

        # Updating the features using feature-level supporting.
        for u in range(len(decoder_to_train.tu)):
            x = decoder_to_train.tu[u](x)
            # aggregate all supporting features
            for supporting_organ in self.support_dict.keys():
                # Feature-level supporting -- Using residual aggregation "1x1 conv followed by addition".

                # Option 1: residual add
                # ### FYI, in Pytorch, you cannot write 'x += y', it will break the gradient and cause leakyReLU issue.
                # e.., x = x + self.support_dict[supporting_organ][u](support_feats[supporting_organ][u])

                # Option 2: channel-wise concatenate
                x = torch.cat((x, self.support_dict[supporting_organ][u](support_feats[supporting_organ][u])), dim=1)
            x = self.to_support[u](x)

            f_skip = skips[-(u + 1 + skips_offset)]
            x_dim = x.size()[2:]
            f_skip_dim = f_skip.size()[2:]
            if x_dim != f_skip_dim:
                x = functional.interpolate(x, size=f_skip_dim, mode='trilinear', align_corners=True)
            x = torch.cat((x, f_skip), dim=1)
            x = decoder_to_train.conv_blocks_localization[u](x)
            seg_outputs.append(decoder_to_train.final_nonlin(decoder_to_train.seg_outputs[u](x)))

        if decoder_to_train.deep_supervision and decoder_to_train.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in zip(list(decoder_to_train.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]
