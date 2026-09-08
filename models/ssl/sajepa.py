import copy

import torch
import torch.nn as nn
from yaml import tokens

from dataset.transform import normalization
from models.ssl.base import BaseSSLModel

from utils.transformer_blocks import Attention, PreNorm, FeedForward, Residual, PositionalEncoding

class TransformerPredictor(nn.Module):
    """Transformer predictor for IJEPA"""
    def __init__(self, hidden_dim, num_layers=4, num_heads=8, dropout=0.1):
        super().__init__()

        # Positional embedding
        self.positional_embedding = PositionalEncoding(hidden_dim= hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(hidden_dim, Attention(hidden_dim, n_heads=num_heads, dropout=dropout))),
                Residual(PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=dropout)))
            )
            for _ in range(num_layers)
        ])

        # Predictor head
        self.predictor_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),  
        )
    
    def forward(self, x):
        # Add positional embedding
        x = self.positional_embedding(x)
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        # Pass through predictor head
        x = self.predictor_head(x)
        return x


class SAJEPAModel(BaseSSLModel):

    def __init__(self, backbone:nn.Module, args_ssl:dict):
        """
        Args:
            backbone (nn.Module): The backbone model for feature extraction.
            args_ssl (dict): A dictionary containing SSL-specific arguments, including:
                - enc_mask_ratio (float): The ratio of the input to be masked for the encoder.
                - pred_mask_ratio (float): The ratio of the input to be masked for the predictor
        """

        super().__init__()

        self.backbone = backbone
        self.args_ssl = args_ssl

        # Model specific parameters
        self.momentum = args_ssl.momentum

        # mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, backbone.hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # predictor (Transformer)
        num_heads = getattr(args_ssl, 'predictor_num_heads', 8)
        num_layers = getattr(args_ssl, 'predictor_num_layers', 4)
        mlp_dim = getattr(args_ssl, 'predictor_mlp_dim', 256)
        dropout = getattr(args_ssl, 'predictor_dropout', 0.1)
        
        self.predictor = TransformerPredictor(
            hidden_dim=backbone.hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # target encoder
        self.target_backbone = copy.deepcopy(backbone)

        for p in self.target_backbone.parameters():
            p.requires_grad = False

    def forward(self, batch):
        # Forward pass through the backbone and predictor
        x_raw = batch["X_raw"] # Raw input data
        x_fold = batch["X_fold"]

        x_raw_norm = normalization.global_z_log_normalization(x=x_raw,stats=self.backbone.stats)
        x_fold_norm = normalization.global_z_log_normalization(x=x_fold,stats=self.backbone.stats)

        # Apply random masking to the input data
        z_context = self.backbone.forward(x_fold_norm)    

        # remove cls
        z_context = z_context[:, 1:, :]

        # Apply predictor to context (SAJEPA like IJEPA)
        z_pred = self.predictor(z_context)

        # Target encoder (sees all patches)
        with torch.no_grad():
            # Target encoder (sees all patches)
            z_target = self.target_backbone.forward(x_raw_norm)
            # remove cls
            z_target = z_target[:, 1:, :]
        
        return {
            "prediction": z_pred,
            "target": z_target,
        }

    def compute_loss(self, outputs, inputs):

        pred = outputs["prediction"]
        target = outputs["target"]

        # Normalize predictions and targets for stable training
        pred = torch.nn.functional.normalize(pred, dim=-1, p=2)
        target = torch.nn.functional.normalize(target, dim=-1, p=2)

        # loss = 2 - 2 * (pred * target).sum(dim=-1).mean()
        loss = nn.functional.smooth_l1_loss(pred, target, reduction='mean')

        return loss
    
    def update_target_encoder(self):
        with torch.no_grad():
            for param_q, param_k in zip(
                self.backbone.parameters(),
                self.target_backbone.parameters()
            ):

                param_k.data.mul_(self.momentum)
                param_k.data.add_((1 - self.momentum) * param_q.data)
