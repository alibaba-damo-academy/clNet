import torch
from torch.nn.utils import prune


def measure_network_sparsity(model, is_prune=False, threshold=1e-6):
    total_weights = 0
    total_near_zero_weights = 0
    try_to_measure = get_paramters_to_prune(model)
    if is_prune:
        for module, param_name in try_to_measure:
            if hasattr(module, "weight_orig"):
                weights = module.weight_orig.data
            else:
                weights = module.weight.data
            num_weights = weights.numel()
            num_near_zero_weights = torch.sum(torch.abs(weights) <= threshold).item()
            total_weights += num_weights
            total_near_zero_weights += num_near_zero_weights
    else:
        for module, param_name in try_to_measure:
            weights = module.weight.data
            num_weights = weights.numel()
            num_near_zero_weights = torch.sum(torch.abs(weights) <= threshold).item()
            total_weights += num_weights
            total_near_zero_weights += num_near_zero_weights
    if total_weights != 0:
        overall_sparsity = total_near_zero_weights / total_weights
        # print(f"Overall sparsity: {overall_sparsity:.4f}")
    else:
        overall_sparsity = None
    return overall_sparsity


def remove_pruning_reparam(decoder_dict):
    for decoder in decoder_dict:
        try_to_remove = get_paramters_to_prune(decoder_dict[decoder])
        if len(try_to_remove) == 0:
            continue
        try:
            for module, param_name in try_to_remove:
                prune.remove(module, param_name)
        except RuntimeError:
            pass


def get_paramters_to_prune(model):
    paramters_ret = []
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            paramters_ret.append((module, "weight"))
    return paramters_ret


def perform_network_initialization_with_pruning_capability(whole_network):
    parameters_to_initialize = []
    for decoder in whole_network.decoder_dict:
        parameters_to_initialize += get_paramters_to_prune(whole_network.decoder_dict[decoder])
    for ema_decoder in whole_network.ema_dict:
        # we did not perform pruning on the encoder.
        if ema_decoder != "general_encoder":
            parameters_to_initialize += get_paramters_to_prune(whole_network.ema_dict[ema_decoder])
    if len(parameters_to_initialize) > 0:
        # prune.global_unstructured(parameters_to_initialize, pruning_method=prune.L1Unstructured, amount=0)
        for param, name in parameters_to_initialize:
            prune.identity(param, name)


def perform_unstructured_lottery_ticket_pruning(model, initial_state_dict, global_sparsity):
    # try to find all linear and conv kernels of the decoding path for pruning.
    parameters_to_prune = get_paramters_to_prune(model)
    # perform the global unstructured pruning
    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=global_sparsity)
    # clone the prune mask and perform "lottery ticket" pruning
    for name, param in model.named_parameters():
        if "weight_orig" in name:
            # original_weight = initial_state_dict[name.replace("_orig", "")]
            original_weight = initial_state_dict[name]
            mask = dict(model.named_buffers())[name.replace("_orig", "_mask")]
            # Reset weights to their initial values, but only for un-pruned connections
            param.data.copy_(original_weight * mask)


def perform_masking_ema_model(model, model_ema):
    prune_mask = {}
    for name, param in model.named_parameters():
        if "_orig" in name:
            mask = dict(model.named_buffers())[name.replace("_orig", "_mask")].clone()
            prune_mask[name] = mask
    for name, param in model_ema.named_parameters():
        if name in prune_mask:
            param.data.copy_(param.data * prune_mask[name])


if __name__ == "__main__":
    in_features, out_features = 100, 10
    linear = torch.nn.Linear(in_features, out_features)
    # for name, param in linear.named_parameters():

    #     print(name, param)

    prune.l1_unstructured(linear, "weight", amount=0.8)

    # for name, param in linear.named_parameters():
    #     print(name, param)
    #
    # for name, param in linear.named_buffers():
    #     print(name, param)
    print(torch.sum(linear.weight_mask)/(in_features * out_features))
    prune.l1_unstructured(linear, "weight", amount=0.8)
    print(torch.sum(linear.weight_mask) / (in_features * out_features))
    prune.l1_unstructured(linear, "weight", amount=0.8)
    print(torch.sum(linear.weight_mask) / (in_features * out_features))
    import numpy as np
    for i in range(10):
        print(np.power(0.9, i))
