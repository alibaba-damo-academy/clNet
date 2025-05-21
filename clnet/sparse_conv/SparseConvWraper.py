from clnet.sparse_conv.utils import *
from clnet.network_architecture.generic_UNet import StackedConvLayers, ConvDropoutNormNonlin
torch.fx.wrap('len')


class ModelSparse(nn.Module):
    def __init__(self, model,
                 sparsity_threshold: float = 1e-3,
                 ignore_list: list = None,
                 subm_conv: bool = False):
        super(ModelSparse, self).__init__()
        traced_model = process_graph(model)
        self.sparsity_threshold = sparsity_threshold
        self.subm_conv = subm_conv
        self.ignore_list = ignore_list if ignore_list is not None else []
        self.model_sparse = self.convert_model_to_sparse(traced_model)

    def convert_layer_to_sparse(self, layer):
        # Replace Conv with SparseConv
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d,
                              spconv.SubMConv1d, spconv.SubMConv2d, spconv.SubMConv3d,
                              spconv.SparseInverseConv1d, spconv.SparseInverseConv2d, spconv.SparseInverseConv3d,
                              nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            return SparseConv(layer, self.sparsity_threshold, self.subm_conv)
        elif isinstance(layer, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
                                nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
            return SparsePool(layer)
        elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            return SparseBatchNorm(layer)
        elif isinstance(layer, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            return SparseInstanceNorm(layer)
        elif isinstance(layer, (nn.ReLU, nn.LeakyReLU, nn.GELU)):
            return SparseNonLinear(layer)
        elif isinstance(layer, nn.Linear):
            return SparseLinear(layer)
        elif isinstance(layer, nn.Flatten):
            return SparseFlatten(layer)
        else:
            return layer

    def convert_seq_to_sparse(self, seq):
        """
        Converts nn.Sequential or nn.ModuleDict into sparse versions.
        """
        if isinstance(seq, (nn.Sequential, nn.ModuleList,
                            StackedConvLayers, ConvDropoutNormNonlin)):
            # Handle nn.Sequential
            sparse_seq = spconv.SparseSequential()
            for name, child in seq.named_children():
                if isinstance(child, (nn.Sequential, nn.ModuleList,
                                      StackedConvLayers, ConvDropoutNormNonlin)):
                    sparse_seq.add_module(name, self.convert_seq_to_sparse(child))
                else:
                    sparse_seq.add_module(name, self.convert_layer_to_sparse(child))
            return sparse_seq
        elif isinstance(seq, nn.ModuleDict):
            # Handle nn.ModuleDict
            sparse_dict = nn.ModuleDict()
            for name, child in seq.items():
                if isinstance(child, (nn.Sequential, nn.ModuleDict, nn.ModuleList,
                                      StackedConvLayers, ConvDropoutNormNonlin)):
                    sparse_dict[name] = self.convert_seq_to_sparse(child)
                else:
                    sparse_dict[name] = self.convert_layer_to_sparse(child)
            return sparse_dict
        else:
            return self.convert_layer_to_sparse(seq)

    def convert_model_to_sparse(self, model):
        """
        Converts the entire model to sparse, handling Sequential, ModuleDict, and individual layers.
        """
        for name, child in model.named_children():
            if name not in self.ignore_list:
                # Recursively convert nested structures
                if isinstance(child, (nn.Sequential, nn.ModuleDict, nn.ModuleList)):
                    setattr(model, name, self.convert_seq_to_sparse(child))
                else:
                    setattr(model, name, self.convert_layer_to_sparse(child))
        return model

    def forward(self, *args, **kwargs):
        out = self.model_sparse(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            if isinstance(out, tuple):
                out = list(out)
            for i in range(len(out)):
                if isinstance(out[i], spconv.SparseConvTensor):
                    out[i] = out[i].dense()
            out = tuple(out)
        elif isinstance(out, spconv.SparseConvTensor):
            return out.dense()
        return out
