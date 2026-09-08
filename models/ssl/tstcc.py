import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataset.transform import normalization
from models.ssl.base import BaseSSLModel

class TSTCCModel(BaseSSLModel):
    """
    TSTCC: Time Series Transformer Contrastive Coding.

    This model learns time series representations by maximizing agreement between
    different augmented views of the same time series using contrastive learning.
    Reference: Eldele et al. "Times-Series Representation Learning via Temporal and Contextual Contrasting (IJCAI 2021)" (AAAI 2021)
    """
    def __init__(self, backbone: nn.Module, args_ssl: dict):
        super(BaseSSLModel, self).__init__()
        
        # Backbone model to extract features from
        self.backbone = backbone
        self.args_ssl = args_ssl

        # Contrastive learning hyperparameters
        self.temperature = getattr(args_ssl, 'temperature', 0.2)
        self.timesteps = getattr(args_ssl, 'timesteps', 4)  # Number of future timesteps to predict in temporal contrasting

        # Temporal Contrasting module
        self.temporal_contrasting = TemporalContrasting(
            num_channels=self.backbone.hidden_dim,
            hidden_dim=self.backbone.hidden_dim,
            timesteps=self.timesteps,
        )

        # Contrastive loss
        self.nt_xent_loss = NTXentLoss(temperature=self.temperature)
    
    def forward(self, batch):
        """
        Forward pass of the TSTCC model.

        Args:
            batch (dict): Input batch containing 'x_raw' tensor

        Returns:
            dict: Output dictionary with:
                - 'z1': Embeddings from first augmented view (batch_size, projection_dim)
                - 'z2': Embeddings from second augmented view (batch_size, projection_dim)
                - 'temporal_contrastive_loss': Temporal contrastive loss
                - 'nt_xent_loss': NT-Xent loss
        """
        x_raw = batch['x_raw']  # (batch_size, seq_length)

        # Data transformation and augmentation
        weak_augmented, strong_augmented = data_transform(x_raw, device=x_raw.device)

        # Feature extraction using backbone
        features_weak = self.backbone(weak_augmented)  # (batch_size, seq_length, hidden_dim)
        features_strong = self.backbone(strong_augmented)  # (batch_size, seq_length, hidden_dim)

        # Normalize features
        features_weak = F.normalize(features_weak, dim=-1)
        features_strong = F.normalize(features_strong, dim=-1)

        return {
            'z1': features_weak,
            'z2': features_strong,
        }

    def compute_loss(self, outputs, inputs, lambda1=1.0, lambda2=0.7):
        """
        Compute the total loss for TSTCC model.

        Args:
            outputs (dict): Output dictionary from forward pass
            inputs (dict): Input batch containing 'x_raw'
        """
        features_weak = outputs['z1']
        features_strong = outputs['z2']

        # Temporal Contrasting
        temporal_loww_weak, projection_weak = self.temporal_contrasting(features_weak, features_strong)
        temporal_loww_strong, projection_strong = self.temporal_contrasting(features_strong, features_weak)

        # Context Contrasting (NT-Xent Loss)
        nt_xent_loss = self.nt_xent_loss(projection_weak, projection_strong)

        loss = lambda1 * (temporal_loww_weak + temporal_loww_strong) + lambda2 * nt_xent_loss
        
        return loss

    
#### Data Transformation and Augmentation for Contrastive Learning
def data_transform(x_raw, device):
    """
    Apply weak and strong augmentations to the raw time series data.
    Args:
        x_raw (torch.Tensor): Raw time series data of shape (batch_size, seq_length).
    Returns:
        tuple: A tuple containing weakly and strongly augmented versions of the input data.
    """
    weak_augmented = scaling(x_raw, device=device)
    strong_augmented = jitter(permutation(x_raw, device=device), device=device)
    return weak_augmented, strong_augmented

def jitter(x_raw, device, sigma=0.8):
    """
    Apply jittering augmentation by adding Gaussian noise to the time series.
    """
    # x_raw: (batch, length)
    noise = torch.normal(mean=0, std=sigma, size=x_raw.size()).to(device)
    return x_raw + noise

def scaling(x_raw, device, sigma=1.1):
    """
    Apply scaling augmentation by multiplying the time series with a random factor.
    """
    # x_raw: (batch, length)
    factor = torch.normal(mean=1.0, std=sigma, size=(x_raw.size(0), 1)).to(device)
    return x_raw * factor

def permutation(x_raw, device, max_segments=5):
    """
    Apply permutation augmentation by dividing the time series into segments and shuffling them.
    """
    # x_raw: (batch, length)
    orig_steps = torch.arange(x_raw.size(1)).to(device)
    num_segs = np.random.randint(1, max_segments, size=(x_raw.size(0),))
    permuted_x = torch.zeros_like(x_raw)
    for i in range(x_raw.size(0)):
        segs = np.array_split(orig_steps.cpu().numpy(), num_segs[i])
        np.random.shuffle(segs)
        permuted_steps = np.concatenate(segs)
        permuted_x[i] = x_raw[i, permuted_steps]

    return permuted_x # permuted_x: (batch, length)

#### Sequential Tranformer for Contrastive Learning
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
 
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
 
 
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
 
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
 
 
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
 
    def forward(self, x):
        return self.net(x)
 
 
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0, "dim doit être divisible par heads"
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))
 
    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = (t.reshape(b, n, h, d // h).transpose(1, 2) for t in (q, k, v))
 
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = attn.softmax(dim=-1)
 
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        return self.to_out(out)
 
 
class Transformer(nn.Module):
    """Pile de `depth` blocs (Attention, FeedForward), en pre-norm + résiduel."""
 
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))),
            ])
            for _ in range(depth)
        ])
 
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
 
 
class SeqTransformer(nn.Module):
    """
    patch_size : dimension d'entrée de chaque pas de temps (= nb de canaux
                 en sortie de l'Encoder, ce que le repo appelle
                 `final_out_channels`)
    dim        : dimension interne du Transformer (= hidden_dim du TC)
    """
 
    def __init__(self, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
 
    def forward(self, forward_seq):
        # forward_seq: (batch, seq_len, patch_dim)
        x = self.patch_to_embedding(forward_seq)
        b = x.shape[0]
        c_tokens = self.c_token.expand(b, -1, -1)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        return x[:, 0]  # état final du token de contexte -> c_t
    
# ----------------------------------------------------------------------
# 4) Temporal Contrasting (module TC du papier)
# ----------------------------------------------------------------------
class TemporalContrasting(nn.Module):
    """
    À partir du passé de la vue A (encodé par le Seq_Transformer en un
    vecteur de contexte c_t), on essaie de prédire les features FUTURES
    de la vue B, un pas de temps à la fois (un `Linear` différent par
    horizon de prédiction, comme dans le papier). La perte est une
    InfoNCE : le futur correspondant doit être identifiable parmi tous
    les futurs des autres échantillons du batch (négatifs).
 
    Renvoie aussi `projection_head(c_t)`, utilisé ensuite pour la
    Contextual Contrasting (NT-Xent).
    """
 
    def __init__(self, num_channels, hidden_dim, timesteps):
        super().__init__()
        self.num_channels = num_channels  # canaux en sortie de l'Encoder
        self.timesteps = timesteps
        # self.device = torch.device
 
        # un Linear par pas de temps futur à prédire (comme `self.Wk` dans TC.py)
        self.Wk = nn.ModuleList(
            [nn.Linear(hidden_dim, num_channels) for _ in range(timesteps)]
        )
 
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, num_channels // 2),
            nn.BatchNorm1d(num_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels // 2, num_channels // 4),
        )
 
        self.seq_transformer = SeqTransformer(
            patch_size=num_channels, dim=hidden_dim, depth=4, heads=4, mlp_dim=64
        )
 
    def forward(self, features_a, features_b):
        """features_a, features_b: (batch, seq_len, hidde)"""
        # z_a = features_a.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        # z_b = features_b.transpose(1, 2)  # (batch, seq_len, hidden_dim)
        z_a = features_a  # (batch, seq_len, hidden_dim)
        z_b = features_b  # (batch, seq_len, hidden_dim)
        batch, seq_len, _ = z_a.shape

        # instant de coupure tiré aléatoirement (comme `t_samples` dans TC.py)
        t = torch.randint(seq_len - self.timesteps, (1,)).item()
 
        # les vrais futurs à deviner, pris dans la vue B
        device = z_a.device
        encode_samples = torch.empty(
            self.timesteps, batch, self.num_channels, device=device
        )
        for i in range(1, self.timesteps + 1):
            encode_samples[i - 1] = z_b[:, t + i, :]
 
        # le Seq_Transformer ne voit que le passé de la vue A (jusqu'à t inclus)
        forward_seq = z_a[:, : t + 1, :]
        c_t = self.seq_transformer(forward_seq)  # (batch, hidden_dim)
 
        nce = 0.0
        for i in range(self.timesteps):
            pred = self.Wk[i](c_t)                       # (batch, num_channels) futur prédit
            total = torch.mm(encode_samples[i], pred.t())  # (batch, batch) similarités
            # softmax sur les colonnes = parmi toutes les prédictions du batch,
            # laquelle correspond à ce vrai futur ? on veut la diagonale.
            nce += F.log_softmax(total, dim=-1).diag().sum()
 
        nce = nce / (-1.0 * batch * self.timesteps)
 
        return nce, self.projection_head(c_t)


#### Contrastive Loss for TSTCC
class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
    This loss is used for contrastive learning, encouraging similar samples to have similar representations.
    """
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature
 
    def forward(self, z1, z2):
        """z1, z2: (batch, dim)"""
        batch = z1.shape[0]
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
 
        representations = torch.cat([z1, z2], dim=0)  # (2B, dim)
        similarity = torch.matmul(representations, representations.t()) / self.temperature
 
        mask = torch.eye(2 * batch, dtype=torch.bool, device=z1.device)
        similarity.masked_fill_(mask, float("-inf"))
 
        positives = torch.arange(2 * batch, device=z1.device)
        positives = (positives + batch) % (2 * batch)
 
        loss = F.cross_entropy(similarity, positives, reduction="sum")
        return loss / (2 * batch)
