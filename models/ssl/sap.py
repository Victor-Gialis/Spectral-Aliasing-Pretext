import torch
import torch.nn as nn
import pandas as pd

from dataset.transform import normalization

from models.backbone.vit1d import ViT1DDecoder
from models.ssl.base import BaseSSLModel

class SAPModel(BaseSSLModel):
    def __init__(self, backbone:nn.Module, args_ssl:dict):
        """
        Spectral Aliasing Pretext (SAP) model for self-supervised learning on time series data.
        Args:
            backbone (nn.Module): Backbone model to extract features from.
            args_ssl (dict): Arguments for self-supervised model.
        """
        super(BaseSSLModel, self).__init__()

        # Backbone model to extract features from
        self.backbone = backbone
        self.args_ssl = args_ssl

        # Downsampling factor for SAP
        self.downsample_factor = args_ssl.downsampling_factor if hasattr(args_ssl, 'downsampling_factor') else 1
        self.symetric_spectrum = args_ssl.symetric_spectrum if hasattr(args_ssl, 'symetric_spectrum') else None

        # Create decoder (force float32)
        self.decoder = ViT1DDecoder(hidden_dim=backbone.hidden_dim).float()
        
        # Mask token for padding when token count mismatch
        hidden_dim = backbone.hidden_dim if hasattr(backbone, 'hidden_dim') else 512
        self.register_buffer('mask_token', torch.zeros(1, 1, hidden_dim, dtype=torch.float32))
        
        # Loss function (force float32)
        self.loss_function = torch.nn.MSELoss().float()
        
        ### Parameters for SAP
        self.loss_domain = args_ssl.loss_domain if hasattr(args_ssl, 'loss_domain') else 'linear'  # Default to linear domain
        self.revin = args_ssl.revin if hasattr(args_ssl, 'revin') else True  # Default to Reversible Instance Normalization (ReVIN)

        print(f"SAPModel initialized with downsample_factor={self.downsample_factor}, symetric_spectrum={self.symetric_spectrum}, loss_domain={self.loss_domain}, revin={self.revin}")
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize weights if necessary
        nn.init.normal_(self.mask_token, std=0.02)
    
    def _pad_embedded_tokens(self, embedded_tokens, target_seq_length):
        """
        Pad embedded_tokens with mask tokens to match target sequence length.
        
        Args:
            embedded_tokens (torch.Tensor): Embedded tokens of shape (batch_size, num_tokens, hidden_dim)
            target_seq_length (int): Target number of tokens (excluding CLS token)
        
        Returns:
            torch.Tensor: Padded embedded tokens of shape (batch_size, num_tokens + num_mask, hidden_dim)
        """
        batch_size, num_tokens, hidden_dim = embedded_tokens.shape
        
        # Account for CLS token if present (first token)
        num_content_tokens = num_tokens - 1  # Exclude CLS token
        
        if num_content_tokens < target_seq_length:
            # Number of mask tokens needed
            num_mask_tokens = target_seq_length - num_content_tokens
            
            # Create mask tokens for this batch
            mask_tokens = self.mask_token.expand(batch_size, num_mask_tokens, hidden_dim)
            
            # Concatenate: CLS token + content tokens + mask tokens
            padded_tokens = torch.cat([embedded_tokens[:, :1, :], 
                                       embedded_tokens[:, 1:, :],
                                       mask_tokens], dim=1)
            return padded_tokens
        
        return embedded_tokens

    def forward(self, batch):
        """
        Forward pass of the SAP model.
        Args:
            batch (dict): Input batch containing 'X_raw' and 'X_folded' tensors.
        Returns:
            dict: Output dictionary with 'prediction' key.
        """
        x_raw = batch['X_raw'] # get raw spectre without fold
        x_fold_scaled = batch['X_folded_scaled'] # get standardized folded spectre
                
        # Forward pass through encoder
        embedded_tokens = self.backbone(x_fold_scaled)
        
        # Pad with mask tokens if necessary
        # x_raw shape: (batch_size, seq_length) -> target tokens = seq_length / patch_size
        batch_size, seq_length = x_raw.shape
        target_num_tokens = seq_length // self.backbone.patch_size
        
        embedded_tokens = self._pad_embedded_tokens(embedded_tokens, target_num_tokens)
        
        # Decoder forward pass
        x_pred_norm = self.decoder(embedded_tokens)

        # RevIN (Reversible Instance Normalization)
        if self.revin:

            if self.backbone.scaling_method == 'normalization':
                x_pred = self.unnormalize(batch, x_pred_norm)

            elif self.backbone.scaling_method == 'standardization':
                x_pred = self.unstandardize(batch, x_pred_norm)

            # Ensure non-negative outputs
            x_pred = super().non_negative_output(x_pred)

        else:
            x_pred = x_pred_norm

        return {"prediction": x_pred}
    
    def compute_loss(self, outputs, inputs):
        targets = inputs['X_raw']
        predictions = outputs['prediction']
        
        # Calculate MSE in chosen domain
        if self.loss_domain == 'linear':
            # Convert from log domain to linear domain
            targets_linear = torch.expm1(targets)
            predictions_linear = torch.expm1(predictions)
            return self.loss_function(predictions_linear, targets_linear)
        
        else:  # 'log' domain (default)
            return self.loss_function(predictions, targets)
