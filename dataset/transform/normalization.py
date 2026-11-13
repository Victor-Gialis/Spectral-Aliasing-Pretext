import torch
from typing import Union, List

def _is_list_of_tensors(x: Union[torch.Tensor, List[torch.Tensor]]) -> bool:
    """Check if x is a list of tensors."""
    return isinstance(x, list) and len(x) > 0 and isinstance(x[0], torch.Tensor)

def instance_min_max_normalization(x: Union[torch.Tensor, List[torch.Tensor]], eps=1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per sample.
    
    Args:
        x: torch.Tensor of shape (batch_size, sequence_length) or List[torch.Tensor]
        eps: small value to avoid division by zero
    
    Returns:
        Normalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x):
        # Process each tensor in the list independently
        return [instance_min_max_normalization(tensor, eps) for tensor in x]
    
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)

    x_norm = (x - x_min) / (x_max - x_min + eps)

    return x_norm

def instance_min_max_unnormalization(x_norm: Union[torch.Tensor, List[torch.Tensor]], stats_from: Union[torch.Tensor, List[torch.Tensor]], eps=1e-8):
    """
    Inverse of instance_min_max_normalization.
    
    Args:
        x_norm: Normalized tensor or list of tensors
        stats_from: Original tensor or list of tensors used for computing stats
        eps: small value to avoid division by zero
    
    Returns:
        Unnormalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x_norm):
        return [instance_min_max_unnormalization(x_n, s_f, eps) for x_n, s_f in zip(x_norm, stats_from)]
    
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin(dim=-1, keepdim=True)
    x_max = stats_log.amax(dim=-1, keepdim=True)

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)
    
    return x_unnorm

def batch_min_max_log_normalization(x: Union[torch.Tensor, List[torch.Tensor]], stats_from: Union[torch.Tensor, List[torch.Tensor]], eps=1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per batch.
    
    Args:
        x: torch.Tensor or List[torch.Tensor]
        stats_from: torch.Tensor or List[torch.Tensor]
        eps: small value to avoid division by zero
    
    Returns:
        Normalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x):
        # For lists, compute global min/max across all tensors in the batch
        all_stats = torch.cat([torch.log1p(s) for s in stats_from])
        x_min = all_stats.amin()
        x_max = all_stats.amax()
        return [(torch.log1p(tensor) - x_min) / (x_max - x_min + eps) for tensor in x]
    
    x_log = torch.log1p(x)
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin()
    x_max = stats_log.amax()

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def batch_min_max_log_unnormalization(x_norm: Union[torch.Tensor, List[torch.Tensor]], stats_from: Union[torch.Tensor, List[torch.Tensor]], eps=1e-8):
    """
    Inverse of batch_min_max_log_normalization.
    
    Args:
        x_norm: Normalized tensor or list of tensors
        stats_from: Original tensor or list of tensors used for computing stats
        eps: small value to avoid division by zero
    
    Returns:
        Unnormalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x_norm):
        all_stats = torch.cat([torch.log1p(s) for s in stats_from])
        x_min = all_stats.amin()
        x_max = all_stats.amax()
        return [torch.expm1(x_n * (x_max - x_min + eps) + x_min) for x_n in x_norm]
    
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin()
    x_max = stats_log.amax()

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)
    
    return x_unnorm

def global_min_max_log_normalization(x: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    
    Args:
        x: torch.Tensor or List[torch.Tensor]
        stats: dict with 'min' and 'max' keys
        eps: small value to avoid division by zero
    
    Returns:
        Normalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x):
        return [global_min_max_log_normalization(tensor, stats, eps) for tensor in x]
    
    x_log = torch.log1p(x)

    x_min = stats["min"]
    x_max = stats["max"]

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def global_min_max_log_unnormalization(x_norm: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Inverse of global_min_max_log_normalization.
    
    Args:
        x_norm: Normalized tensor or list of tensors
        stats: dict with 'min' and 'max' keys
        eps: small value to avoid division by zero
    
    Returns:
        Unnormalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x_norm):
        return [global_min_max_log_unnormalization(x_n, stats, eps) for x_n in x_norm]
    
    x_min = stats["min"]
    x_max = stats["max"]

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)

    return x_unnorm

def global_z_log_normalization(x: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Log + mean_std normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    
    Args:
        x: torch.Tensor or List[torch.Tensor]
        stats: dict with 'mean' and 'std' keys
        eps: small value to avoid division by zero
    
    Returns:
        Normalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x):
        return [global_z_log_normalization(tensor, stats, eps) for tensor in x]
    
    x_log = torch.log1p(x)
    
    x_mean= stats["mean"]
    x_std = stats["std"]

    x_norm = (x_log - x_mean) / (x_std + eps)

    return x_norm

def global_z_log_unnormalization(x_norm: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Inverse of global_z_log_normalization.
    
    Args:
        x_norm: Normalized tensor or list of tensors
        stats: dict with 'mean' and 'std' keys
        eps: small value to avoid division by zero
    
    Returns:
        Unnormalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x_norm):
        return [global_z_log_unnormalization(x_n, stats, eps) for x_n in x_norm]
    
    x_mean = stats["mean"]
    x_std = stats["std"]

    x_log = x_norm * (x_std + eps) + x_mean
    x_unnorm = torch.expm1(x_log)

    return x_unnorm

def global_z_score_normalization(x: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Mean-std normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    
    Args:
        x: torch.Tensor or List[torch.Tensor]
        stats: dict with 'mean' and 'std' keys
        eps: small value to avoid division by zero
    
    Returns:
        Normalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x):
        return [global_z_score_normalization(tensor, stats, eps) for tensor in x]
    
    x_mean= stats["mean"]
    x_std = stats["std"]

    x_norm = (x - x_mean) / (x_std + eps)

    return x_norm

def global_z_score_unnormalization(x_norm: Union[torch.Tensor, List[torch.Tensor]], stats: dict, eps: float = 1e-8):
    """
    Inverse of global_z_score_normalization.
    
    Args:
        x_norm: Normalized tensor or list of tensors
        stats: dict with 'mean' and 'std' keys
        eps: small value to avoid division by zero
    
    Returns:
        Unnormalized tensor or list of tensors with same structure as input
    """
    if _is_list_of_tensors(x_norm):
        return [global_z_score_unnormalization(x_n, stats, eps) for x_n in x_norm]
    
    x_mean = stats["mean"]
    x_std = stats["std"]

    x_unnorm = x_norm * (x_std + eps) + x_mean

    return x_unnorm