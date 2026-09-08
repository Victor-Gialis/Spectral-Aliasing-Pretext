import torch
import torch.nn as nn

from dataset.transform import normalization
from models.backbone.vit1d import ViT1DDecoder
from models.ssl.base import BaseSSLModel

class MAEModel(BaseSSLModel):
    def __init__(self, backbone:nn.Module, args_ssl:dict):
        """
        Masked Autoencoder (MAE) model for self-supervised learning on time series data.
        Args:
            patch_size (int): Size of each patch for tokenization.
            hidden_dim (int): Dimension of the hidden embeddings.
            n_layers (int): Number of transformer layers in encoder and decoder.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            mask_ratio (float): Ratio of tokens to mask during training.
        """
        super(BaseSSLModel, self).__init__()

        # Backbone model to extract features from 
        self.backbone = backbone
        self.args_ssl = args_ssl

        # Mask token
        self.mask_ratio = args_ssl.mask_ratio #tokens masking ratio of mae
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.backbone.hidden_dim))

        # Create pretrain head
        self.decoder = ViT1DDecoder()

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize weights if necessary
        nn.init.normal_(self.mask_token, std=0.02)
    
    def forward(self, batch):
        """
        Forward pass of the MAE model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        Returns:
            dict: A dictionary containing the reconstructed output and the mask.
        """
        x_raw = batch['X_raw']

        # Normalization
        x_norm = normalization.global_z_log_normalization(x=x_raw, stats=self.backbone.stats)

        # Tokenisation
        input_tokens = self.backbone.forward_tokeniser(x_norm)
        b, n, _ = input_tokens.shape

        # Apply random masking
        input_tokens_masked, mask, ids_restore = self.random_masking(input_tokens, mask_ratio=self.mask_ratio)
        
        # Encode visible tokens
        embedded_tokens = self.backbone.forward_encoder(input_tokens_masked)

        # Prepare decoder input by appending mask tokens
        len_keep = input_tokens_masked.shape[1]
        mask_tokens = self.mask_token.repeat(b, n - len_keep, 1)
        
        embedded_tokens_ = torch.cat([embedded_tokens[:, 1:, :], mask_tokens], dim=1) # exclude cls token
        embedded_tokens_ = torch.gather(embedded_tokens_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, embedded_tokens_.size(-1)))
        embedded_tokens_ = torch.cat([embedded_tokens[:, :1, :], embedded_tokens_], dim=1) # add back cls token
        
        # Decode to reconstruct original input
        x_pred_norm = self.decoder(embedded_tokens_)

        # Unnormalize
        x_pred = normalization.global_z_log_unnormalization(x_norm=x_pred_norm, stats=self.backbone.stats)

        # Ensure non-negative outputs
        x_pred = super().non_negative_output(x_pred)

        # Restore mask in the original X shape
        # mask = mask.unsqueeze(-1).expand(-1, -1, x_pred.shape[-1]).reshape(b, -1)

        return {"prediction":x_pred, "mask":mask}

    def compute_loss(self, outputs, inputs):
        targets = inputs['X_raw']
        predictions = outputs['prediction']
        mask = outputs['mask']
        
        # Rearrange targets to match prediction shape
        b, l = targets.shape
        predictions = predictions.reshape(b, l // self.backbone.patch_size, self.backbone.patch_size)
        targets = targets.reshape(b, l // self.backbone.patch_size, self.backbone.patch_size)
        
        # Compute loss only on masked tokens
        loss = ((predictions - targets) ** 2)
        loss = (loss * mask.unsqueeze(-1)).sum() / mask.sum()
        
        return loss
    
    def random_masking(self, x, mask_ratio):
        """
        Perform random masking on the embded tokens.
        Args:
            x (torch.Tensor): Input tokens of shape (batch_size, num_tokens, embed_dim).
            mask_ratio (float): Ratio of tokens to mask.
        Returns:
            x_masked (torch.Tensor): Masked input tokens.
            mask (torch.Tensor): Binary mask indicating which tokens were masked.
            ids_restore (torch.Tensor): Indices to restore the original token order.
        """
        b, n, _ = x.shape

        len_keep = int(n * (1 - mask_ratio))
        noise = torch.rand(b, n, device=x.device)
        
        # interpret noise as random shuffle indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        # get the masked input
        x_masked = x.gather(dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.size(-1)))
        
        # create the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([b, n], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    