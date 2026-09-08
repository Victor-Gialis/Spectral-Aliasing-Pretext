import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.transform import normalization
from models.ssl.base import BaseSSLModel

class TS2VecModel(BaseSSLModel):
    """
    TS2Vec: Towards Universal Representation of Time Series.
    
    This model learns time series representations by maximizing agreement between 
    different augmented views of the same time series using contrastive learning.
    
    Reference: Yue et al. "TS2Vec: Towards Universal Representation of Time Series" (AAAI 2022)
    """
    
    def __init__(self, backbone: nn.Module, args_ssl: dict):
        """
        Initialize TS2Vec model.
        
        Args:
            backbone (nn.Module): Backbone encoder to extract features from.
            args_ssl (dict): Arguments for self-supervised model.
                - temperature (float): Temperature parameter for contrastive loss. Default: 0.1
                - use_cosine_similarity (bool): Use cosine similarity instead of dot product. Default: True
                - temporal_unit (int): Temporal unit for instance discrimination. Default: 0 (no temporal masking)
        """
        super(BaseSSLModel, self).__init__()
        
        # Backbone model to extract features from
        self.backbone = backbone
        self.args_ssl = args_ssl
        
        # Contrastive learning hyperparameters
        self.temperature = getattr(args_ssl, 'temperature', 0.1)
        self.use_cosine_similarity = getattr(args_ssl, 'use_cosine_similarity', True)
        self.temporal_unit = getattr(args_ssl, 'temporal_unit', 0)
        self.projection_dim = getattr(args_ssl, 'projection_dim', 256)
        
        # Loss function
        self.hierarchical_contrastive_loss = HierarchicalContrastiveLoss(temperature=self.temperature)
            
    def generate_binomial_mask(self, x_raw, p=0.5):
        mask = torch.from_numpy(np.random.binomial(1, p, size=x_raw.shape))
        return mask
    
    def forward(self, batch):
        """
        Forward pass of the TS2Vec model.
        
        Args:
            batch (dict): Input batch containing 'x_raw' tensor
        
        Returns:
            dict: Output dictionary with:
                - 'z1': Embeddings from first augmented view (batch_size, projection_dim)
                - 'z2': Embeddings from second augmented view (batch_size, projection_dim)
        """
        # Get raw time series from batch
        x_raw = batch['x_raw']  # (batch_size, seq_length)
        x_raw = x_raw.unsqueeze(-1) if x_raw.ndim == 2 else x_raw  # Ensure x_raw has shape (batch_size, seq_length, input_dim)
        ts_lenght = x_raw.shape[1]

        # Apply masking to create two augmented views of the time series
        mask = self.generate_binomial_mask(x_raw)
        x_raw[~mask] = 0  # Apply mask to the raw time series

        crop_length = np.random.randint(low=2**(self.temporal_unit + 1), high=ts_lenght+1)
        crop_left = np.random.randint(ts_lenght - crop_length + 1)
        crop_right = crop_left + crop_length

        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_lenght + 1)

        crop_offset = np.random.randint(low=-crop_left, high=ts_lenght - crop_length + 1)
        
        # Create two different augmented views with bounds checking
        all_idx1 = (crop_offset + crop_eleft) + np.arange(crop_right - crop_eleft)
        all_idx1 = np.clip(all_idx1, 0, ts_lenght - 1)  # Ensure indices are within bounds
        x_aug1 = x_raw[:, all_idx1]  # First augmented view
        x_aug1 = x_aug1[:, -crop_length:] if x_aug1.shape[1] >= crop_length else x_aug1  # Ensure crop_length
        
        all_idx2 = (crop_offset + crop_left) + np.arange(crop_eright - crop_left)
        all_idx2 = np.clip(all_idx2, 0, ts_lenght - 1)  # Ensure indices are within bounds
        x_aug2 = x_raw[:, all_idx2]  # Second augmented view
        x_aug2 = x_aug2[:, :crop_length] if x_aug2.shape[1] >= crop_length else x_aug2  # Ensure crop_length

        # # Normalize augmented views
        x_norm1 = normalization.global_z_score_normalization(x=x_aug1, stats=self.backbone.stats)
        x_norm2 = normalization.global_z_score_normalization(x=x_aug2, stats=self.backbone.stats)

        # Encode both views
        # backbone returns embeddings of shape (batch_size, num_tokens, hidden_dim)
        z1 = self.backbone.forward(x_norm1)  # (batch_size, num_tokens, hidden_dim)
        z2 = self.backbone.forward(x_norm2)   # (batch_size, num_tokens, hidden_dim)
        
        return {"z1": z1, "z2": z2}
    
    def compute_loss(self, outputs, inputs):
        """
        Compute contrastive loss between two augmented views.
        
        Args:
            outputs (dict): Model outputs containing 'z1' and 'z2'
            inputs (dict): Input batch (not used but kept for compatibility)
        
        Returns:
            torch.Tensor: Contrastive loss
        """
        z1 = outputs["z1"]  # (batch_size, projection_dim)
        z2 = outputs["z2"]  # (batch_size, projection_dim)
        
        # Compute NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        loss = self.hierarchical_contrastive_loss(z1, z2)
        
        return loss


class HierarchicalContrastiveLoss(nn.Module):
    """
    TS2Vec Hierarchical Contrastive Loss.
    
    Combines instance-level and temporal-level contrasts at multiple scales.
    """
    
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, alpha=0.5, temporal_unit=0):
        """
        Args:
            z1, z2: (batch_size, time_size, embedding_dim)
            alpha: weight for instance contrastive loss
            temporal_unit: minimum scale for temporal contrasts
        """
        loss = torch.tensor(0.0, device=z1.device)
        d = 0

        while z1.size(1) > 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            if d >= temporal_unit:
                loss += self.temporal_contrastive_loss(z1, z2)
            d += 1
            z1 = nn.functional.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = nn.functional.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

        if z1.size(1) == 1:
            if alpha != 0:
                loss += alpha * self.instance_contrastive_loss(z1, z2)
            d += 1
                
        return loss / d
    
    def instance_contrastive_loss(self, z1, z2):
        """
        Instance-level contrastive loss (Formula 2 in TS2Vec paper).
        
        For each timestamp t, compare series samples between two views.
        
        Args:
            z1, z2: (batch_size, time_size, embedding_dim)
        
        Returns:
            loss: scalar
        """
        B, T = z1.size(0), z1.size(1)
        
        if B == 1:
            return z1.new_tensor(0.0)

        z = torch.cat([z1, z2], dim=0)       # (2B, T, C)
        z = z.transpose(0, 1)                # (T, 2B, C)
        sim = torch.matmul(z, z.transpose(1, 2))   # (T, 2B, 2B)

        # Extract off-diagonal elements using triangular matrices
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        # Positive pairs: (i, B+i-1) and (B+i, i)
        i = torch.arange(B, device=z1.device)
        loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
        
        return loss

    def temporal_contrastive_loss(self, z1, z2):
        """
        Temporal contrastive loss (Formula 1 in TS2Vec paper).
        
        For each series sample, compare timestamps between two views.
        
        Args:
            z1, z2: (batch_size, time_size, embedding_dim)
        
        Returns:
            loss: scalar
        """
        B, T = z1.size(0), z1.size(1)
        
        if T == 1:
            return z1.new_tensor(0.0)

        z = torch.cat([z1, z2], dim=1)       # (B, 2T, C)
        sim = torch.matmul(z, z.transpose(1, 2))   # (B, 2T, 2T)

        # Extract off-diagonal elements using triangular matrices
        logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
        logits += torch.triu(sim, diagonal=1)[:, :, 1:]
        logits = -F.log_softmax(logits, dim=-1)

        # Positive pairs: (t, T+t-1) and (T+t, t)
        t = torch.arange(T, device=z1.device)
        loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
        
        return loss