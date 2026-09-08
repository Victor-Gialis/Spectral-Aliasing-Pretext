import torch
import torch.nn as nn

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

        # Create decoder
        self.decoder = ViT1DDecoder()
        
        # Mask token for padding when token count mismatch
        hidden_dim = backbone.hidden_dim if hasattr(backbone, 'hidden_dim') else 512
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Loss function
        self.loss_function = torch.nn.MSELoss()

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
        x_fold = batch['X_folded'] # get folded spectrum in input
        
        # Normalization
        x_norm = normalization.global_z_log_normalization(x=x_fold, stats=self.backbone.stats)
        
        # Forward pass through encoder
        embedded_tokens = self.backbone(x_norm)
        
        # Pad with mask tokens if necessary
        # x_raw shape: (batch_size, seq_length) -> target tokens = seq_length / patch_size
        batch_size, seq_length = x_raw.shape
        target_num_tokens = seq_length // self.backbone.patch_size
        
        embedded_tokens = self._pad_embedded_tokens(embedded_tokens, target_num_tokens)
        
        # Decoder forward pass
        x_pred_norm = self.decoder(embedded_tokens)
        
        # Unnormalize
        x_pred = normalization.global_z_log_unnormalization(x_norm=x_pred_norm, stats=self.backbone.stats)

        # Ensure non-negative outputs
        x_pred = super().non_negative_output(x_pred)

        return {"prediction": x_pred}
    
    def compute_loss(self, outputs, inputs):
        targets = inputs['X_raw']
        predictions = outputs['prediction']
        return self.loss_function(predictions, targets)