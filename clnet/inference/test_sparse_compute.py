import torch
from spconv import pytorch as spconv
import torch.nn as nn


torch.manual_seed(42)


def convert_dense_layer_to_sparse(dense_layer):
    """
    Converts a single dense layer (e.g., nn.Conv3d) to its sparse counterpart.
    """
    if isinstance(dense_layer, nn.Conv3d):
        sparse_conv = spconv.SparseConv3d(
            in_channels=dense_layer.in_channels,
            out_channels=dense_layer.out_channels,
            kernel_size=dense_layer.kernel_size,
            stride=dense_layer.stride,
            padding=dense_layer.padding,
            bias=dense_layer.bias is not None,
        )
        # print("sparse", sparse_conv.weight.data.shape)
        # print("dense", dense_layer.weight.data.shape)
        sparse_conv.weight.data = dense_layer.weight.permute(0, 2, 3, 4, 1).contiguous().clone()
        # print(dense_layer.weight)
        if dense_layer.bias is not None:
            sparse_conv.bias.data = dense_layer.bias.contiguous().clone()
            # print(dense_layer.bias)
        return sparse_conv
    elif isinstance(dense_layer, nn.ConvTranspose3d):
        sparse_conv = spconv.SparseConvTranspose3d(
            in_channels=dense_layer.in_channels,
            out_channels=dense_layer.out_channels,
            kernel_size=dense_layer.kernel_size,
            stride=dense_layer.stride,
            padding=dense_layer.padding,
            bias=dense_layer.bias is not None,
        )
        # print(sparse_conv.weight.data.shape)
        # print(dense_layer.weight.data.shape)
        sparse_conv.weight.data = dense_layer.weight.permute(1, 2, 3, 4, 0).contiguous().clone()
        if dense_layer.bias is not None:
            sparse_conv.bias.data = dense_layer.bias.contiguous().clone()
        return sparse_conv
    elif isinstance(dense_layer, nn.MaxPool3d):
        return spconv.SparseMaxPool3d(
            kernel_size=dense_layer.kernel_size,
            stride=dense_layer.stride,
            padding=dense_layer.padding,
        )
    elif isinstance(dense_layer, nn.Sequential):
        sparse_layers = [convert_dense_layer_to_sparse(layer) for layer in dense_layer]
        return sparse_layers
    else:
        return dense_layer  # For non-convolution layers, return as-is


def convert_model_to_sparse(dense_model):
    """
    Converts a dense 3D network to a sparse 3D network with the same structure.
    Args:
        dense_model (nn.Module): The original dense model.
    Returns:
        sparse_model (nn.Module): The corresponding sparse model.
    """
    class SparseModel(nn.Module):
        def __init__(self, dense_model):
            super(SparseModel, self).__init__()
            # Convert each dense layer to sparse
            layers = []
            for name, layer in dense_model.named_children():
                layers += convert_dense_layer_to_sparse(layer)
            self.toDense = spconv.ToDense()
            self.layers = spconv.SparseSequential(*layers)

        def forward(self, x):

            return self.toDense(self.layers(x))

    return SparseModel(dense_model)


class DenseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(),
            # nn.Conv3d(4, out_channels, kernel_size=1),
        )

    def forward(self, x):
        return self.layers(x)


class SparseNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SparseNet, self).__init__()
        self.net = spconv.SparseSequential(
            spconv.SparseConv3d(1, 32, 3, 1),
            nn.ReLU(),
            spconv.SparseConv3d(32, 64, 3, 1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        # x: [N, 28, 28, 1], must be NHWC tensor
        x_sp = spconv.SparseConvTensor.from_dense(x.reshape(-1, 32, 32, 32, 1))
        # create SparseConvTensor manually: see SparseConvTensor.from_dense
        x = self.net(x_sp)
        x = spconv.ToDense(x)

        return x


def copy_weights(dense_block, sparse_block):
    for dense_layer, sparse_layer in zip(dense_block, sparse_block):
        if isinstance(dense_layer, nn.Conv3d) and isinstance(sparse_layer, spconv.SparseConv3d):
            sparse_layer.weight.data = dense_layer.weight.data.permute(1, 2, 3, 4, 0).clone()
            sparse_layer.bias.data = dense_layer.bias.data.clone()


in_chan = 2

# Create a dense dynamic model
dense_model = DenseNet(in_channels=in_chan, out_channels=16).to("cuda").eval()
# Convert to sparse model
sparse_model = convert_model_to_sparse(dense_model).to("cuda").eval()
# sparse_model = SparseNet(in_channels=in_chan, out_channels=16).to("cuda").eval()


# Dense input
dense_input = 1e5*torch.randn(1, in_chan, 32, 32, 32).cuda()  # Dense input tensor
dense_input_ndhwc = dense_input.permute(0, 2, 3, 4, 1).contiguous()
sparse_input = spconv.SparseConvTensor.from_dense(dense_input_ndhwc)
# dense_input = 1e10 * torch.randn(1, in_chan, 32, 32, 32).cuda()
# dense_input_ndhwc = dense_input.permute(0, 2, 3, 4, 1).contiguous()
# sparse_input = spconv.SparseConvTensor.from_dense(dense_input_ndhwc)

# Forward pass
dense_output = dense_model(dense_input)
sparse_output = sparse_model(sparse_input)

print(torch.abs(dense_output - sparse_output).sum())
print("Dense Output Shape:", dense_output.shape)
print("Sparse Output Features Shape:", sparse_output.shape)
print(torch.allclose(dense_output, sparse_output, atol=1e-5))

# import torch
# import spconv.pytorch as spconv
# from spconv.pytorch import SparseConvTensor
#
# # Dense Conv3D
# dense_conv = torch.nn.Conv3d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
#
# # SparseConv3D
# sparse_conv = spconv.SparseConv3d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
#
# # Weight transformation
# dense_weights = dense_conv.weight.data  # (16, 8, 3, 3, 3)
# sparse_weights = dense_weights.permute(0, 2, 3, 4, 1).contiguous()  # (16, 3, 3, 3, 8)
# sparse_conv.weight.data = sparse_weights.clone()
#
# # Input tensors
# input_dense = torch.randn(1, 8, 10, 10, 10)  # Shape: (B, C, D, H, W)
#
# # Get non-zero positions and values
# non_zero_mask = input_dense != 0  # Boolean mask for non-zero values
# indices = torch.nonzero(non_zero_mask)  # Non-zero indices (flattened view)
#
# # Extract features corresponding to non-zero positions
# features = input_dense[non_zero_mask]  # Features from non-zero positions
#
# # Define the spatial shape
# spatial_shape = input_dense.shape[2:]  # (D, H, W)
#
# # Create SparseConvTensor
# input_sparse = SparseConvTensor(
#     features=features,         # Non-zero values
#     indices=indices.long(),    # Coordinates of non-zero values
#     spatial_shape=spatial_shape,  # Spatial dimensions
#     batch_size=input_dense.size(0)  # Batch size
# )
#
# # Outputs
# output_dense = dense_conv(input_dense)  # Dense output
# output_sparse = sparse_conv(input_sparse).dense()  # Sparse output
#
# # Validate outputs
# print(torch.allclose(output_dense, output_sparse, atol=1e-5))
