from scipy import stats
import torch 
import torch.nn as nn
import pandas as pd

class BaseSSLModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("Forward method not implemented!")
    
    def compute_loss(self, predictions, targets):
        """
        Compute the loss between predictions and targets.
        Args:
            predictions (torch.Tensor): Model predictions.
            targets (torch.Tensor): Ground truth targets.
        Returns:
            torch.Tensor: Computed loss.
        """
        raise NotImplementedError("Compute_loss method not implemented!")
    
    def non_negative_output(self, x):
        """
        Ensure that the output tensor is non-negative.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Non-negative output tensor.
        """
        return torch.clamp(x, min=0.0)

    def get_statistics(self, batch):
        stats = pd.DataFrame([s[self.backbone.domain] for s in batch["stats"]])
        return stats

    def unnormalize(self, batch, x_norm):
        """
        Unnormalize the input tensor using the provided statistics.
        Args:
            x_norm (torch.Tensor): Normalized input tensor.
            stats (pd.DataFrame): DataFrame containing 'min' and 'max' values for unnormalization.
        Returns:
            torch.Tensor: Unnormalized tensor.
        """
        stats = self.get_statistics(batch)

        min_val = torch.tensor(stats['min'].values, dtype=torch.float32).to(x_norm.device)
        max_val = torch.tensor(stats['max'].values, dtype=torch.float32).to(x_norm.device)
        
        # Reshape min and max to match x_norm dimensions
        min_val = min_val.view(-1, 1)
        max_val = max_val.view(-1, 1)
        
        x_unnorm = x_norm * (max_val - min_val) + min_val
        return x_unnorm

    def unstandardize(self, batch, x_std):
        """
        Unstandardize the input tensor using the provided statistics.
        Args:
            x_std (torch.Tensor): Standardized input tensor.
            stats (pd.DataFrame): DataFrame containing 'mean' and 'std' values for unstandardization.
        Returns:
            torch.Tensor: Unstandardized tensor.
        """
        stats = self.get_statistics(batch)
        
        mean_val = torch.tensor(stats['mean'].values, dtype=torch.float32).to(x_std.device)
        std_val = torch.tensor(stats['std'].values, dtype=torch.float32).to(x_std.device)
        
        # Reshape mean and std to match x_std dimensions
        mean_val = mean_val.view(-1, 1)
        std_val = std_val.view(-1, 1)
        
        x_unstd = x_std * std_val + mean_val
        return x_unstd

    @classmethod
    def from_config(cls, config:dict):
        raise NotImplementedError("From Config method not implemented")