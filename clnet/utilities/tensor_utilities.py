import torch


def sum_tensor(inp, axes, keepdim=False):
    axes = torch.unique(torch.tensor(axes, dtype=torch.int32))
    sorted_axes, _ = torch.sort(axes, descending=True)
    if keepdim:
        for ax in axes:
            inp = torch.sum(inp, dim=int(ax), keepdim=True)
    else:
        for ax in sorted_axes:
            inp = torch.sum(inp, dim=int(ax))
    return inp


def mean_tensor(inp, axes, keepdim=False):
    axes = torch.unique(torch.tensor(axes, dtype=torch.int32))
    sorted_axes, _ = torch.sort(axes, descending=True)
    if keepdim:
        for ax in axes:
            inp = torch.mean(inp, dim=int(ax), keepdim=True)
    else:
        # Sort axes in reverse to ensure reduction happens correctly
        for ax in sorted_axes:
            inp = torch.mean(inp, dim=int(ax))
    return inp


def flip(x, dim):
    """
    flips the tensor at dimension dim (mirroring!)
    :param x:
    :param dim:
    :return:
    """
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]


