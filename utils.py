import torch


def module_device(module):
    return next(module.parameters()).device


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask
