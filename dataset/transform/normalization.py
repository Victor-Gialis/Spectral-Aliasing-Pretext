import torch

def instance_min_max_log_normalization(x:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per sample.
    """
    x_log = torch.log1p(x)
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin(dim=-1, keepdim=True)
    x_max = stats_log.amax(dim=-1, keepdim=True)

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def instance_min_max_log_unnormalization(x_norm:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin(dim=-1, keepdim=True)
    x_max = stats_log.amax(dim=-1, keepdim=True)

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)
    
    return x_unnorm

def batch_min_max_log_normalization(x:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per batch.
    """
    x_log = torch.log1p(x)
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin()
    x_max = stats_log.amax()

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def batch_min_max_log_unnormalization(x_norm:torch.Tensor, stats_from:torch.Tensor, eps=1e-8):
    stats_log = torch.log1p(stats_from)

    x_min = stats_log.amin()
    x_max = stats_log.amax()

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)
    
    return x_unnorm

def global_min_max_log_normalization(x: torch.Tensor, stats: dict, eps: float = 1e-8):
    """
    Log + min-max normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    """
    x_log = torch.log1p(x)

    x_min = stats["min"]
    x_max = stats["max"]

    x_norm = (x_log - x_min) / (x_max - x_min + eps)

    return x_norm

def global_min_max_log_unnormalization(x_norm: torch.Tensor, stats: dict, eps: float = 1e-8):
    x_min = stats["min"]
    x_max = stats["max"]

    x_log = x_norm * (x_max - x_min + eps) + x_min
    x_unnorm = torch.expm1(x_log)

    return x_unnorm

def global_z_log_normalization(x: torch.Tensor, stats: dict, eps: float = 1e-8):
    """
    Log + mean_std normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    """
    x_log = torch.log1p(x)
    
    x_mean= stats["mean"]
    x_std = stats["std"]

    x_norm = (x_log - x_mean) / (x_std + eps)

    return x_norm

def global_z_log_unnormalization(x_norm: torch.Tensor, stats: dict, eps: float = 1e-8):
    x_mean = stats["mean"]
    x_std = stats["std"]

    x_log = x_norm * (x_std + eps) + x_mean
    x_unnorm = torch.expm1(x_log)

    return x_unnorm

def global_z_score_normalization(x: torch.Tensor, stats: dict, eps: float = 1e-8):
    """
    Mean-std normalization using stats_from for consistent scaling.
    Normalization is done per by the global min-max train dataset 
    """
    
    x_mean= stats["mean"]
    x_std = stats["std"]

    x_norm = (x - x_mean) / (x_std + eps)

    return x_norm

def global_z_score_unnormalization(x_norm: torch.Tensor, stats: dict, eps: float = 1e-8):
    x_mean = stats["mean"]
    x_std = stats["std"]

    x_unnorm = x_norm * (x_std + eps) + x_mean

    return x_unnorm