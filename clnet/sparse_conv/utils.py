import os
import psutil

import torch
from torch import nn
from torch.fx import symbolic_trace

import spconv.pytorch as spconv


def maybe_to_sparse(x, sp_th=None):
    if not isinstance(x, spconv.SparseConvTensor):
        if sp_th is not None and isinstance(sp_th, (int, float)):
            x[torch.abs(x) < sp_th] = 0
        dims = x.dim()
        tensor_perm_order = [0] + list(range(2, dims)) + [1]
        x = spconv.SparseConvTensor.from_dense(x.permute(*tensor_perm_order))
    return x


class SparseConv(spconv.SparseModule):
    def __init__(self, child, sparsity_threshold=1e-3, subm_conv=False):
        super(SparseConv, self).__init__()
        self.sparsity_threshold = sparsity_threshold

        if isinstance(child, (spconv.SparseConv1d, spconv.SparseConv2d, spconv.SparseConv3d,
                              spconv.SubMConv1d, spconv.SubMConv2d, spconv.SubMConv3d,
                              spconv.SparseInverseConv1d, spconv.SparseInverseConv2d, spconv.SparseInverseConv3d,
                              spconv.SparseConvTranspose1d, spconv.SparseConvTranspose2d, spconv.SparseConvTranspose3d)):
            self.sparse_conv = child
        elif isinstance(child, nn.Conv1d):
            if subm_conv:
                self.sparse_conv = spconv.SubMConv1d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            else:
                self.sparse_conv = spconv.SparseConv1d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(0, 2, 1).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)
        elif isinstance(child, nn.Conv2d):
            if subm_conv:
                self.sparse_conv = spconv.SubMConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            else:
                self.sparse_conv = spconv.SparseConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(0, 2, 3, 1).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)

        elif isinstance(child, nn.Conv3d):
            if subm_conv:
                self.sparse_conv = spconv.SubMConv3d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            else:
                self.sparse_conv = spconv.SparseConv3d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    bias=(child.bias is not None)
                )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(0, 2, 3, 4, 1).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)
        elif isinstance(child, nn.ConvTranspose1d):
            self.sparse_conv = spconv.SparseConvTranspose1d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                bias=(child.bias is not None)
            )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(1, 2, 0).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)
        elif isinstance(child, nn.ConvTranspose2d):
            self.sparse_conv = spconv.SparseConvTranspose2d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                bias=(child.bias is not None)
            )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(1, 2, 3, 0).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)
        elif isinstance(child, nn.ConvTranspose3d):
            self.sparse_conv = spconv.SparseConvTranspose3d(
                in_channels=child.in_channels,
                out_channels=child.out_channels,
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation,
                bias=(child.bias is not None)
            )
            # Copy weights and sparsify
            with torch.no_grad():
                self.sparse_conv.weight.data.copy_(child.weight.data.permute(1, 2, 3, 4, 0).contiguous())
                self.sparse_conv.weight.data[torch.abs(self.sparse_conv.weight.data) < sparsity_threshold] = 0
                if child.bias is not None:
                    self.sparse_conv.bias.data.copy_(child.bias.data)

    def forward(self, x):
        x = maybe_to_sparse(x, self.sparsity_threshold)
        autocast_enabled = torch.is_autocast_enabled()
        tensor_dtype = torch.float16 if autocast_enabled else torch.float32
        if x.features.dtype != tensor_dtype:
            x = x.replace_feature(x.features.to(tensor_dtype))
        if autocast_enabled:
            with torch.cuda.amp.autocast():
                ret = self.sparse_conv(x)
                return ret
            # self.sparse_conv = self.sparse_conv.half()
        ret = self.sparse_conv(x)
        return ret


class SparsePool(spconv.SparseModule):
    def __init__(self, child):
        super(SparsePool, self).__init__()
        if isinstance(child, (spconv.SparseMaxPool2d, spconv.SparseMaxPool3d, spconv.SparseAvgPool2d, spconv.SparseAvgPool3d)):
            self.pooling = child
        elif isinstance(child, nn.MaxPool2d):
            self.pooling = spconv.SparseMaxPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation)
        elif isinstance(child, nn.MaxPool3d):
            self.pooling = spconv.SparseMaxPool3d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation)
        elif isinstance(child, nn.AvgPool2d):
            self.pooling = spconv.SparseAvgPool2d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation)
        elif isinstance(child, nn.AvgPool3d):
            self.pooling = spconv.SparseAvgPool3d(
                kernel_size=child.kernel_size,
                stride=child.stride,
                padding=child.padding,
                dilation=child.dilation)

    def forward(self, x):
        x = maybe_to_sparse(x)
        return self.pooling(x)


class SparseNonLinear(spconv.SparseModule):
    # we try to apply ReLU to eliminate all negative values (it hurts performance)
    def __init__(self, child, keep_default_relu=False):
        super(SparseNonLinear, self).__init__()
        if keep_default_relu:
            self.relu = child
        else:
            self.relu = nn.ReLU()

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(self.relu.forward(x.features))
        return self.relu.forward(x)


class SparseBatchNorm(spconv.SparseModule):
    def __init__(self, child):
        super(SparseBatchNorm, self).__init__()
        if child.weight is not None and child.bias is not None:
            flag_affine = True
        else:
            flag_affine = False

        self.norm_layer = nn.BatchNorm1d(child.num_features, affine=flag_affine, momentum=child.momentum)
        if flag_affine:
            self.norm_layer.weight.data = child.weight.data.clone()
            self.norm_layer.bias.data = child.bias.data.clone()
        self.norm_layer.momentum = child.momentum

    def forward(self, x):
        x = maybe_to_sparse(x)
        return x.replace_feature(self.norm_layer.forward(x.features))


class SparseInstanceNorm(spconv.SparseModule):
    """
    Sparse Instance Normalization for spconv.SparseConvTensor.
    This normalizes the features of a sparse tensor per instance and channel.
    """
    def __init__(self, child):
        """
        We neglect the `momentum`, `running_mean`, `running_var`, and `track_running_stats` parameters
        for simplicity and computation efficiency.
        """
        super(SparseInstanceNorm, self).__init__()
        self.eps = child.eps
        self.affine = child.affine
        if self.affine:
            self.weight = child.weight.data.clone()  # Scale (gamma)
            self.bias = child.bias.data.clone()  # Shift (beta)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        x = maybe_to_sparse(x)

        # Determine if we are in an autocast context
        autocast_enabled = torch.is_autocast_enabled()

        # Extract features and batch indices
        features = x.features  # Do not cast here; handle later
        indices = x.indices[:, 0].to(torch.int64)  # Extract batch indices (shape: [N])
        spatial_size = torch.tensor(x.spatial_size, device=features.device, dtype=torch.float32)  # Keep spatial size in float32

        # Number of instances (batch size) and channels
        num_instances = indices.max().item() + 1  # Total number of instances
        num_channels = features.size(1)

        # Create tensors to store sum, squared sum, and counts for each instance and channel
        tensor_dtype = torch.float16 if autocast_enabled else torch.float32
        instance_channel_sum = torch.zeros((num_instances, num_channels), device=features.device, dtype=tensor_dtype)
        instance_channel_square_sum = torch.zeros((num_instances, num_channels), device=features.device, dtype=tensor_dtype)
        instance_channel_counts = torch.full(
            (num_instances, 1), fill_value=spatial_size, device=features.device, dtype=torch.float32
        )

        # Perform scatter_add operations
        features = features.to(tensor_dtype) if autocast_enabled else features.to(torch.float32)
        instance_channel_sum.scatter_add_(0, indices.unsqueeze(-1).expand_as(features), features)
        instance_channel_square_sum.scatter_add_(0, indices.unsqueeze(-1).expand_as(features), features.mul(features))

        # Compute per-instance mean and variance in float32 for stability
        if autocast_enabled:
            instance_channel_sum = instance_channel_sum.to(dtype=torch.float32)
            instance_channel_square_sum = instance_channel_square_sum.to(dtype=torch.float32)

        instance_channel_mean = instance_channel_sum / instance_channel_counts
        instance_channel_var = (instance_channel_square_sum / instance_channel_counts) - instance_channel_mean.mul(instance_channel_mean)

        # Expand the mean and variance for each feature
        expanded_mean = instance_channel_mean[indices]
        expanded_var = instance_channel_var[indices]

        # Normalize the features
        normalized_features = (features - expanded_mean.to(tensor_dtype)) / torch.sqrt(expanded_var.to(tensor_dtype) + self.eps)

        # Apply affine transformation if enabled
        if self.affine:
            normalized_features = normalized_features * self.weight.to(dtype=tensor_dtype, device=features.device) \
                                  + self.bias.to(dtype=tensor_dtype, device=features.device)

        if autocast_enabled:
            if not self.training:
                normalized_features = normalized_features.to(dtype=torch.float32)
            with torch.cuda.amp.autocast():
                ret = x.replace_feature(normalized_features)
                return ret
        else:
            ret = x.replace_feature(normalized_features)
            return ret


class SparseFlatten(nn.Module):
    def __init__(self, child):
        super(SparseFlatten, self).__init__()
        self.flatten_layer = child

    def forward(self, x):
        # Convert sparse input to dense
        if isinstance(x, spconv.SparseConvTensor):
            x = x.dense()
        # Apply InstanceNorm3d on dense data
        return self.flatten_layer(x)


class SparseLinear(nn.Module):
    def __init__(self, child):
        super(SparseLinear, self).__init__()
        self.linear = child

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            x = x.dense()
        return self.linear(x)


class ToDense(nn.Module):
    def forward(self, x):
        return x.dense() if isinstance(x, spconv.SparseConvTensor) else x


def sparse_cat(tensors, dim=None):
    autocast_enabled = torch.is_autocast_enabled()
    if dim is None or dim == 1:
        if len(tensors) == 0:
            raise ValueError("The input list of tensors is empty.")
        tensor_dtype = torch.float16 if autocast_enabled else torch.float32
        # Combine all indices and features
        all_indices = torch.cat([tensor.indices for tensor in tensors], dim=0)
        all_features = torch.cat([tensor.features for tensor in tensors], dim=0)

        # Find unique indices and map features to them
        unique_indices, inverse_indices = torch.unique(all_indices, dim=0, return_inverse=True)

        # Compute the total feature dimension
        feature_dims = [tensor.features.size(1) for tensor in tensors]
        total_feature_dim = sum(feature_dims)

        # Create aligned feature tensor
        aligned_features = \
            torch.zeros((unique_indices.size(0), total_feature_dim),
                        device=all_features.device, dtype=tensor_dtype)

        # Scatter features into aligned_features based on inverse_indices
        feature_offsets = torch.cumsum(torch.tensor([0] + feature_dims[:-1], device=all_features.device), dim=0)
        for offset, tensor in zip(feature_offsets, tensors):
            mask = inverse_indices[:tensor.indices.size(0)]
            aligned_features[mask, offset:offset + tensor.features.size(1)] = tensor.features.to(tensor_dtype)
            inverse_indices = inverse_indices[tensor.indices.size(0):]
        # Create concatenated SparseConvTensor

        return spconv.SparseConvTensor(
            features=aligned_features,
            indices=unique_indices,
            spatial_shape=tensors[0].spatial_shape,
            batch_size=tensors[0].batch_size
        )
    else:
        raise ValueError(f"Concatenation along dimension {dim} is not supported for SparseConvTensor.")


def restore_module_structure(traced, original_model, parent_name=""):
    """
    Recursively restores the module structure (e.g., ModuleDict, Sequential) to the traced model.
    """
    for name, module in original_model.named_children():
        # Create full name for nested submodules
        full_name = f"{parent_name}.{name}" if parent_name else name

        # If it's a ModuleDict, restore its structure by adding submodules one by one
        if isinstance(module, nn.ModuleDict):
            # Restore each item in the ModuleDict
            for sub_name, sub_module in module.items():
                restore_module_structure(traced, sub_module, full_name + "." + sub_name)
            # Add the ModuleDict itself
            traced.add_submodule(full_name, module)

        # If it's a Sequential, add it directly as a submodule
        elif isinstance(module, nn.Sequential):
            traced.add_submodule(full_name, module)
            # Recursively restore each submodule inside Sequential
            for sub_name, sub_module in module.named_children():
                restore_module_structure(traced, sub_module, full_name + "." + sub_name)

        # For other types of modules, add them directly as submodules
        else:
            traced.add_submodule(full_name, module)

    return traced


def process_graph(model):
    """
    Trace the given module and add a ToDense layer before torch.cat and torch.flatten operations.
    """
    try:
        # "symbolic_trace" will convert all module structures to "nn.Module".
        traced = symbolic_trace(model)
    except Exception:
        # If tracing fails, return the module as is
        return model

    # Modify the graph if necessary
    flag_to_dense = False
    flag_sparse_cat = False
    for node in traced.graph.nodes:
        if node.op == "call_function" and \
                node.target in {torch.cat, torch.concat, torch.concatenate,
                                torch.flatten, torch.log, torch.log_softmax,
                                torch.sigmoid}:
            flag_to_dense = True
            if node.target in {torch.flatten}:
                with traced.graph.inserting_before(node):
                    to_dense_node = traced.graph.create_node(
                        "call_module", "toDense", args=(node.args[0],), kwargs={}
                    )
                # Update the `flatten` node's input
                node.args = (to_dense_node,) + node.args[1:]
            elif node.target in {torch.cat, torch.concat, torch.concatenate}:
                flag_sparse_cat = True
                node.target = sparse_cat
            else:
                new_args = []
                for arg in node.args[0]:  # Iterate over tensors passed to the operation
                    with traced.graph.inserting_before(node):
                        to_dense_node = traced.graph.create_node(
                            "call_module", "toDense", args=(arg,), kwargs={}
                        )
                    new_args.append(to_dense_node)
                # Update the original node's inputs
                node.args = (tuple(new_args),) + node.args[1:]
    if flag_sparse_cat:
        traced.sparse_channel_cat = sparse_cat
    #  add Dense module
    if not hasattr(traced, "toDense") and flag_to_dense:
        traced.add_module("toDense", ToDense())
    else:
        return model

    traced = restore_module_structure(traced, model)

    # Recompile the modified graph
    traced.graph.lint()
    traced.recompile()
    # del traced.graph

    return traced


def log_memory_usage(message=""):
    """Logs the current memory usage of the process."""
    process = psutil.Process(os.getpid())  # Get the current process
    mem_info = process.memory_info()
    print(f"{message} - RSS: {mem_info.rss / (1024 ** 2):.2f} MB, VMS: {mem_info.vms / (1024 ** 2):.2f} MB")


def calculate_sparsity(model, threshold=1e-5):
    """
    Calculates the sparsity of a PyTorch model, considering a threshold.

    Args:
        model (nn.Module): The PyTorch model to measure.
        threshold (float): Values with absolute magnitude less than this are considered zero.

    Returns:
        float: The overall sparsity percentage.
        dict: Layer-wise sparsity information.
    """
    total_params = 0
    near_zero_params = 0
    layer_sparsity = {}

    for name, param in model.named_parameters():
        if param.requires_grad:  # Only consider trainable parameters
            num_params = param.numel()
            num_near_zeros = (param.abs() < threshold).sum().item()

            total_params += num_params
            near_zero_params += num_near_zeros

            # Record layer sparsity
            layer_sparsity[name] = {
                "total_params": num_params,
                "near_zero_params": num_near_zeros,
                "sparsity_percent": 100.0 * num_near_zeros / num_params
            }

    overall_sparsity = 100.0 * near_zero_params / total_params if total_params > 0 else 0.0
    return overall_sparsity, layer_sparsity


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    # Input tensor
    N, C, D, H, W = 4, 4, 10, 10, 10
    input_tensor1 = torch.randn(N, C, D, H, W)
    input_tensor2 = torch.randn(N, C, D, H, W)
    input_tensor1[input_tensor1 < 0] = 0
    input_tensor2[input_tensor1 > 0] = 0

    concat_tensor = torch.cat((input_tensor1, input_tensor2), 1)
    perm_order = [0, 2, 3, 4, 1]
    sparse_tensor1 = spconv.SparseConvTensor.from_dense(input_tensor1.permute(*perm_order))
    sparse_tensor2 = spconv.SparseConvTensor.from_dense(input_tensor2.permute(*perm_order))
    sparse_concat_tensor = sparse_cat((sparse_tensor1, sparse_tensor2), 1)
    sparse_concat_tensor_to_dense = sparse_concat_tensor.dense()
    print(torch.allclose(concat_tensor, sparse_concat_tensor_to_dense, atol=1e-6))
