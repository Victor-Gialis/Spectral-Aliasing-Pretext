import torch
import torch.nn as nn
from dataset.transform import normalization
from models.backbone.spectral_position import spectral_patch_coordinates
from models.backbone.vit1d import ViT1DEncoder
from models.backbone.cnn import DilatedCNNEncoder, CNNEncoder

class DownstreamModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        head: nn.Module,
        freeze_backbone: bool = False,
        device:  torch.device | str = "cpu",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        
        # Log dtypes
        backbone_dtype = next(self.backbone.parameters()).dtype
        head_dtype = next(self.head.parameters()).dtype
        print(f"\n✓Model scaling method: {self.backbone.scaling_method}")  
        print(f"\n✓Backbone dtype: {backbone_dtype}, Head dtype: {head_dtype}")

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, batch, get_attention=False):
        """
        batch: dict (standardisé par ton dataloader)
        """
        if self.backbone.domain == "time":
            if self.backbone.scaling_method is not None:
                # x = batch['x_raw_scaled']  # (batch_size, seq_length, num_channels)
                x = batch['x_folded_scaled']  # (batch_size, seq_length, num_channels)
            else:
                # x = batch['x_raw']
                x = batch['x_folded']  # (batch_size, seq_length, num_channels)

        elif self.backbone.domain == "frequency":
            if self.backbone.scaling_method is not None:
                # x = batch['X_raw_scaled']  # (batch_size, seq_length, num_channels)
                x = batch['X_folded_scaled']  # (batch_size, seq_length, num_channels)
            else:
                # x = batch['X_raw']  # (batch_size, seq_length, num_channels)
                x = batch['X_folded']  # (batch_size, seq_length, num_channels)

        else:
            raise ValueError(f"Backbone domain {self.backbone.domain} not recognized. Please use 'time' or 'frequency'.")

        # Get head dtype for conversion after backbone
        head_dtype = next(self.head.parameters()).dtype

        if isinstance(self.backbone, DilatedCNNEncoder)or isinstance(self.backbone, CNNEncoder):
            # If the backbone is a CNN-based encoder, we can directly use the output features for classification.
            # x_norm: (batch_size, seq_length, num_channels)
            features = self.backbone(x)
            aggregated = features.mean(dim=1)  # (batch_size, hidden_dim)
            # Convert to head dtype for computation
            aggregated = aggregated.to(head_dtype)
            outputs = self.head(aggregated)

            return outputs

        elif isinstance(self.backbone, ViT1DEncoder):
            # If the backbone is a ViT-based encoder, we can use the class token for classification.
            # x_norm: (batch_size, seq_length, num_channels)
            # Pass through the ViT encoder
            # The ViT encoder will return the class token and the attention scores
            positions = None
            if self.backbone.frequency_positions:
                positions = spectral_patch_coordinates(batch, self.backbone.patch_size, folded=True)
            features = self.backbone(x, positions=positions)
            cls_token = features[:,0]
            # Convert to head dtype for computation
            cls_token = cls_token.to(head_dtype)
            outputs = self.head(cls_token)

            if get_attention:
                attention_scores = self.backbone.get_attention_scores(x, positions=positions)
                return outputs, attention_scores

            else:
                return outputs

    def compute_loss(self, outputs, batch):
        return self.head.compute_loss(outputs, batch)
