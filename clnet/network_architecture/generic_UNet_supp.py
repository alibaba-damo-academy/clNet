import torch
from torch import nn
import torch.nn.functional
from clnet.network_architecture.initialization import InitWeights_He
from clnet.network_architecture.generic_UNet import ConvDropoutNormNonlin
from clnet.network_architecture.center_crop import center_crop_feature_pairs


class Generic_UNet_Supporting_Organ(nn.Module):
    def __init__(self, decoder_dict, train_supporting_dict, decoder, num_pool, conv_op=nn.Conv3d, norm_op=nn.InstanceNorm3d,
                 norm_op_kwargs=None, dropout_op=nn.Dropout3d, dropout_op_kwargs=None, nonlin=nn.LeakyReLU, nonlin_kwargs=None,
                 supporting_weight=0.5, weight_initializer=InitWeights_He(1e-2)):
        super(Generic_UNet_Supporting_Organ, self).__init__()
        self.weight_initializer = weight_initializer
        self.decoder = decoder
        self.decoder_dict = decoder_dict
        self.decoder_to_train = decoder_dict[decoder]
        self.supporting_weight = supporting_weight

        self.num_pool = num_pool
        self.to_support = []
        self.dim_mismatch = True
        self.support_dict = train_supporting_dict[decoder]
        conv_kwargs_1x1 = {'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 1, 'bias': True}
        conv_kwargs_3x3 = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        for u in range(num_pool):
            nfeatures_supporting_organ = decoder_dict[decoder].seg_outputs[u].out_channels
            nfeatures_to_support_organ = decoder_dict[decoder].seg_outputs[u].out_channels
            for supporting_organ in train_supporting_dict[decoder]:
                nfeatures_supporting_organ += decoder_dict[supporting_organ].seg_outputs[u].out_channels
            self.to_support.append(nn.Sequential(
                ConvDropoutNormNonlin(nfeatures_supporting_organ, 2 * nfeatures_supporting_organ,
                                      conv_op, conv_kwargs_3x3, norm_op, norm_op_kwargs, dropout_op,
                                      dropout_op_kwargs, nonlin, nonlin_kwargs),
                ConvDropoutNormNonlin(2 * nfeatures_supporting_organ, nfeatures_to_support_organ,
                                      conv_op, conv_kwargs_1x1, None, None, None, None, None, None)
            )
            )
        self.to_support = nn.ModuleList(self.to_support)
        self.support_gate = nn.Parameter(torch.tensor(supporting_weight))
        if self.weight_initializer is not None:
            self.apply(self.weight_initializer)

    def forward(self, skips, x):
        support_feats = {}
        seg_output = []
        # Get the supporting features from the supporting decoders
        for supporting_organ in self.support_dict:
            support_feats[supporting_organ] = self.decoder_dict[supporting_organ](skips, x)
        decoder_to_train_output = list(self.decoder_to_train(skips, x))
        # Get the features from the decoder to train and center crop the supporting features
        for u in range(self.num_pool):
            to_support = decoder_to_train_output[u]
            for supporting_organ in support_feats:
                to_support, supporting = center_crop_feature_pairs(to_support, support_feats[supporting_organ][u])
                to_support = torch.cat((to_support, supporting), dim=1)
            # Combine the supporting features with the features from the decoder to train.
            # We try to preserve the original decoder's accuracy, s.t., the support_gate is learnable (it could be 0 if supporting is not helpful).
            seg_output.append(self.to_support[u](to_support) * self.support_gate + decoder_to_train_output[u])

        return tuple(seg_output)
