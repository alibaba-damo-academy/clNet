def center_crop_features(tensor, target_shape):
    dims = tensor.dim() - 2  # number of spatial dimensions

    if dims == 2 and len(target_shape) == 2:
        # Fast path for 2D: tensor shape (B, C, H, W)
        target_h, target_w = target_shape
        _, _, h, w = tensor.size()
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return tensor[:, :, start_h:start_h + target_h, start_w:start_w + target_w]

    elif dims == 3 and len(target_shape) == 3:
        # Fast path for 3D: tensor shape (B, C, D, H, W)
        target_d, target_h, target_w = target_shape
        _, _, d, h, w = tensor.size()
        start_d = (d - target_d) // 2
        start_h = (h - target_h) // 2
        start_w = (w - target_w) // 2
        return tensor[:, :, start_d:start_d + target_d, start_h:start_h + target_h, start_w:start_w + target_w]

    else:
        # Generic fallback for arbitrary spatial dimensions
        current_shape = tensor.shape[2:]
        start_indices = [(curr - tgt) // 2 for curr, tgt in zip(current_shape, target_shape)]
        slices = [slice(None), slice(None)]  # for batch and channel dimensions
        for start, tgt in zip(start_indices, target_shape):
            slices.append(slice(start, start + tgt))
        return tensor[tuple(slices)]


def center_crop_feature_pairs(tensor1, tensor2):
    # Get spatial dimensions (assumes dimensions beyond the first two are spatial)
    spatial1 = tensor1.shape[2:]
    spatial2 = tensor2.shape[2:]

    # Determine the target shape as the minimum size in each spatial dimension
    target_shape = tuple(min(s1, s2) for s1, s2 in zip(spatial1, spatial2))

    # Apply center crop to both tensors
    if spatial1 != target_shape:
        tensor1 = center_crop_features(tensor1, target_shape)
    if spatial2 != target_shape:
        tensor2 = center_crop_features(tensor2, target_shape)

    return tensor1, tensor2
