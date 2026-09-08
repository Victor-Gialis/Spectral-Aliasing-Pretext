import math
import torch
import torch.nn as nn
from torch import Tensor
from einops.layers.torch import Rearrange

# Patch Embedding
# This class is used to convert the input image into patches and then flatten them.  
class Tokeniser(nn.Module):
    def __init__(self, patch_size = 8, embed_dim = 128):
        """
        Args:
            patch_size (int): Size of the patches to be extracted from the input image.
            emb_size (int): Size of the embedding vector for each patch.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.projection = nn.Sequential(
            # Reshape the input tensor to create patches
            Rearrange('b (n p) -> b n p', p = patch_size), # Rearrange the input tensor to create patches
            nn.Linear(patch_size, embed_dim) # Linear projection of the patch to the embedding size
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.projection(x)
        return x

# Multi-head Self-Attention
# This is a simplified version of the multi-head self-attention mechanism used in transformers.
class Attention(nn.Module):
    def __init__(self, dim, n_heads=8, dropout=0.):
        """
        Args: 
            dim (int): Dimension of the input features.
            heads (int): Number of attention heads.
            dropout (float): Dropout rate.
        """
         # Initialize the parent class
        super().__init__()
        self.heads = n_heads
        self.dim = dim
        self.scale = self.dim ** -0.5

        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)  
        self.value = nn.Linear(dim, dim, bias=False)

        self.last_attn = None

    def forward(self, x):
        # x : (batch, seq_len, dim)
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_output, attn_output_weights = self.attention(q, k, v)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        self.last_attn = attn.detach()

        return attn_output

# Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.ln = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.ln(x))

# Feed Forward
class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

# Residual Connection
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return x + self.fn(x, *args, **kwargs)

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Pré-calcul du buffer
        pe = torch.zeros(1, max_seq_len, hidden_dim, dtype=torch.float32)
        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2, dtype=torch.float32)* -(math.log(10000.0) / hidden_dim))
        
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pos_encoding", pe, persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x):
        """
        Renvoie un positional encoding pour une séquence de longueur n.
        """
        _,n,_ = x.shape
        if n > self.max_seq_len:
            raise ValueError(f"n ({n}) dépasse max_seq_len ({self.max_seq_len})")
        return x + self.pos_encoding[:, :n, :]

class Predictor(nn.Module):
    def __init__(self, hidden_dim: int, predictor_hidden_dim: int):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, predictor_hidden_dim),
            nn.BatchNorm1d(predictor_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_hidden_dim, hidden_dim)
        )

    def forward(self, x):
        return self.predictor(x)