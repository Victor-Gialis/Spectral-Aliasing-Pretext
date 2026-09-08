import torch
import torch.nn as nn
from dataset.transform import normalization
from models.backbone.vit1d import ViT1DDecoder
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

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, batch, get_attention=False):
        """
        batch: dict (standardisé par ton dataloader)
        """
        if self.backbone.domain == "time":
            x_raw = batch['x_raw']  # (batch_size, seq_length, num_channels)
            
        elif self.backbone.domain == "frequency":
            x_raw = batch['X_raw']  # (batch_size, seq_length, num_channels)

        # If stats is not empty dict, normalize the input data using global z-log normalization
        if self.backbone.stats and self.backbone.normalization is not None:

            if self.backbone.normalization == "z-score-log":
                # Z-score followed by log transformation
                x_norm = normalization.global_z_log_normalization(x=x_raw, stats=self.backbone.stats)
            
            elif self.backbone.normalization == "z-score":
                # Z-score normalization only
                x_norm = normalization.global_z_score_normalization(x=x_raw, stats=self.backbone.stats)
            
            else :
                raise ValueError(f"Normalization method {self.backbone.normalization} not recognized. Please use 'z-score-log' or 'z-score'.")

        else:
            x_norm = x_raw  # If stats is empty, use raw data without normalization

        if isinstance(self.backbone, DilatedCNNEncoder)or isinstance(self.backbone, CNNEncoder):
            # If the backbone is a CNN-based encoder, we can directly use the output features for classification.
            # x_norm: (batch_size, seq_length, num_channels)
            features = self.backbone(x_norm)
            aggregated = features.mean(dim=1)  # (batch_size, hidden_dim)
            outputs = self.head(aggregated)

            return outputs

        elif isinstance(self.backbone, ViT1DDecoder):
            # If the backbone is a ViT-based encoder, we can use the class token for classification.
            # x_norm: (batch_size, seq_length, num_channels)
            # Pass through the ViT encoder
            # The ViT encoder will return the class token and the attention scores
            features = self.backbone(x_norm)
            cls_token = features[:,0]
            outputs = self.head(cls_token)

            if get_attention:
                attention_scores = self.backbone.get_attention_scores(x_norm)
                return outputs, attention_scores

            else:
                return outputs

    def compute_loss(self, outputs, batch):
        return self.head.compute_loss(outputs, batch)
