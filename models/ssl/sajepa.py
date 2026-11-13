import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.transform import normalization
from models.ssl.base import BaseSSLModel

from utils.transformer_blocks import Attention, PreNorm, FeedForward, Residual, PositionalEncoding

class TransformerPredictor(nn.Module):
    """Transformer predictor for SA-JEPA"""
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
            args_ssl (dict): empty
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
        dropout = getattr(args_ssl, 'predictor_dropout', 0.1)
        
        self.predictor = TransformerPredictor(
            hidden_dim=backbone.hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )

        # Context encoder 
        # The context_encoder = backbone 
    
        # Tarjet encoder
        self.target_backbone = copy.deepcopy(backbone)

        # Freeze target encoder parameters
        for p in self.target_backbone.parameters():
            p.requires_grad = False

        # JEPA loss
        # self.jepa_loss = nn.SmoothL1Loss()

        # # VICReg loss
        self.vicReg_loss = VICReg(
            hidden_dim=backbone.hidden_dim,
            output_dim=getattr(args_ssl, 'vicreg_output_dim', 8176),
            inv_coeff=getattr(args_ssl, 'vicreg_inv_coeff', 25.0),
            var_coeff=getattr(args_ssl, 'vicreg_var_coeff', 25.0),
            cov_coeff=getattr(args_ssl, 'vicreg_cov_coeff', 1.0),
            var_threshold=getattr(args_ssl, 'vicreg_var_threshold', 1.0)
        )

    def _load_momentum_scheduler(self, momentum_scheduler):
        """
        Load a momentum scheduler for the target encoder.
        Args:
            momentum_scheduler: A list or callable that provides the momentum value for each training step.
        """
        self.momentum_scheduler = momentum_scheduler

    def forward(self, batch):
        x_raw_norm = batch["X_raw"]
        x_fold_norm = batch["X_folded"]

        # Step 1 :Foward through backbone to get contextualized embeddings for the folded input
        tokens = self.backbone.forward_tokeniser(x_fold_norm)

        ### Add mask tokens if the number of tokens is less than the number of patches
        if tokens.shape[1] < (x_raw_norm.shape[1] // self.backbone.patch_size):
            num_mask_tokens = x_raw_norm.shape[1] // self.backbone.patch_size - tokens.shape[1]
            mask_tokens = self.mask_token.repeat(tokens.shape[0], num_mask_tokens, 1)
            tokens = torch.cat([tokens, mask_tokens], dim=1)

        z_context = self.backbone.forward_encoder(tokens) # Contextualized embeddings from the backbone
        z_context = z_context[:, 1:, :]  # remove cls
        z_pred = self.predictor(z_context) # Apply predictor to context (SAJEPA like IJEPA)

        # Step 2 : Forward through target encoder to get target embeddings for the raw input
        with torch.no_grad():
            z_target = self.target_backbone.forward(x_raw_norm) # Target encoder (sees all patches)
            z_target = z_target[:, 1:, :]  # remove cls
            z_target = z_target.detach() # Apply stop-gradient to prevent collapse (BYOL/SimSiam style)

        return {
            "prediction": z_pred,
            "target": z_target,
        }

    def compute_loss(self, outputs, inputs):
        """
        Compute loss using VICReg regularization
        Args:
            outputs: Dictionary containing model outputs (prediction and target embeddings)
            inputs: Dictionary containing input data (not used in this loss computation)
        Returns:
            loss: Computed VICReg loss (invariance, variance, covariance)
        """
        pred = outputs["prediction"]
        target = outputs["target"]

        # Compute I-JEPA loss (Smooth L1 Loss)
        # loss = self.jepa_loss(pred, target)

        # # Reshape from (batch_size, num_tokens, dim) to (batch_size * num_tokens, dim)
        # batch_size, num_tokens, dim = pred.shape
        # pred = pred.reshape(-1, dim)
        # target = target.reshape(-1, dim)

        # # Compute VICReg loss
        loss = self.vicReg_loss(pred, target)
        
        return loss
    
    def update_target_encoder(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_context, param_target in zip(self.backbone.parameters(),self.target_backbone.parameters()):
                param_target.data.mul_(m) # Scale target encoder parameters by momentum
                param_target.data.add_((1 - m) * param_context.data) # Update target encoder parameters with weighted average of context encoder parameters


class Expanders(nn.Module):
    """Expander module to project embeddings to a higher dimensional space"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        layers = nn.ModuleList()
        f = list(map(int, [hidden_dim, output_dim]))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.BatchNorm1d(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1]), biais=False)
        self.expander = nn.Sequential(*layers)


    def forward(self, x):
        for layer in self.expander:
            x = layer(x)
        return x
class VICReg(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=8176, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, var_threshold=1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.var_threshold = var_threshold

        # Create an expander to project embeddings to a higher dimensional space
        self.expander = Expanders(hidden_dim, output_dim)

    def foward(self, pred, target):
        pred = self.expander(pred)
        target = self.expander(target)
        N, D = pred.shape

        # 1. Invariance : espace brut, comme VICReg
        inv_loss = F.mse_loss(pred, target)

        # 2. Variance : uniquement sur pred (target est détaché)
        pred_c = pred - pred.mean(dim=0)
        std = torch.sqrt(pred_c.var(dim=0) + 1e-4)
        var_loss = F.relu(self.var_threshold - std).mean()

        # 3. Covariance : uniquement sur pred
        cov = (pred_c.T @ pred_c) / (N - 1)
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D

        loss = self.inv_coeff * inv_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss

        # Logs pour détecter un collapse (std ≈ 0 => collapse)
        with torch.no_grad():
            t_std = torch.sqrt(target.var(dim=0) + 1e-4)
            self.last_logs = {
                "inv": inv_loss.detach(),
                "var": var_loss.detach(),
                "cov": cov_loss.detach(),
                "pred_std": std.mean().detach(),
                "target_std": t_std.mean().detach(),
            }
        return loss