import torch
import torch.nn as nn
from types import SimpleNamespace
from utils.transformer_blocks import Tokeniser, Attention, PreNorm, FeedForward, Residual, PositionalEncoding

class ViT1DEncoder(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=512, n_layers=3, heads=8, dropout=0.2565, domain="time", normalization=None):
        super().__init__()
        # Encoder hyperparameters
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout
        self.domain = domain
        self.normalization = normalization

        # Pretrain dataset Stats
        self.stats = dict()

        # Tokeniser
        self.tokeniser = Tokeniser(patch_size=patch_size, embed_dim=hidden_dim)

        # Positional embedding
        self.positional_embedding = PositionalEncoding(hidden_dim= self.hidden_dim)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        # Encoder parameters
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(hidden_dim, Attention(hidden_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.encoder_output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize cls token
        nn.init.normal_(self.cls_token, std=0.02)

        # Using Xavier uniform initialization for linear layers
        for layer in self.encoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward_tokeniser(self, x):
        return self.tokeniser(x)
    
    def forward_encoder(self, x):
        """Encodes the input using the encoder."""
        # Extract batch size and number of tokens
        b = x.shape[0]
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)  # Étendre le cls_token pour correspondre à la taille du batch
        x = torch.cat([cls_tokens, x], dim=1)  # Concaténer le cls_token avec les embeddings des patches
        # Add positional embedding
        x = self.positional_embedding(x)
        # Encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        # Encoder output
        encoded_tokens = self.encoder_output_head(x)
        return encoded_tokens
    
    def forward(self, x):
        tokens = self.forward_tokeniser(x)
        embeded_tokens = self.forward_encoder(tokens)
        return embeded_tokens 
    
    def _loads_stats(self, stats):
        self.stats = stats

    def _get_arguments(self):
        """
        Return the arguments of the model.
        Returns:
            args (SimpleNamespace): Arguments of the model
        """
        return SimpleNamespace(
            patch_size = self.patch_size,
            hidden_dim = self.hidden_dim,
            n_layers = self.n_layers,
            heads = self.heads,
            dropout = self.dropout,
            )
    
    def get_attention_scores(self, x, layer_idx=0):
        """Returns attention scores from a specific encoder layer.
        
        Args:
            x: Input tensor
            layer_idx: Layer index (default: -1 for last layer, 0 for first, etc.)
        
        Returns:
            Attention scores from cls token to all patches, shape (batch, n_patches)
        """
        # Tokeniser
        x = self.tokeniser(x)
        # Extract batch size and number of tokens
        b, _, _ = x.shape
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add positional embedding
        x = self.positional_embedding(x)
        
        # Collect all attention modules
        attention_modules = []

        for layer in self.encoder_layers:
            residual_block = layer[0]
            prenorm = residual_block.fn
            attention_module = prenorm.fn
            attention_modules.append(attention_module)
        
        # Select the desired layer
        if len(attention_modules) == 0:
            return None
        
        selected_attention = attention_modules[layer_idx]
        
        # Pass through encoder layers
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            
            # If this is the selected layer, extract attention scores
            if i == layer_idx or (layer_idx == -1 and i == len(self.encoder_layers) - 1):
                if hasattr(selected_attention, 'last_attn'):
                    attn_scores = selected_attention.last_attn[:, 0, 1:]
                    return attn_scores
        
        return None
    
    def get_attention_rollout(self, x):
        """Returns attention rollout scores from cls token to all patches.
        
        Rollout accumulates attention across all layers to capture how attention
        propagates through the entire network from input to output.
        
        Args:
            x: Input tensor
        
        Returns:
            Rollout attention scores from cls token to all patches, shape (batch, n_patches)
        """
        # Tokeniser
        x = self.tokeniser(x)
        # Extract batch size and number of tokens
        b, n_patches, _ = x.shape
        n_tokens = n_patches + 1  # +1 for cls token
        
        # Add cls token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add positional embedding
        x = self.positional_embedding(x)
        
        # Initialize rollout matrix (identity matrix)
        I = torch.eye(n_tokens).unsqueeze(0).expand(b, -1, -1).to(x.device)
        rollout = I.clone()  # Start with identity for rollout
        
        # Pass through encoder layers and accumulate attention
        for layer in self.encoder_layers:
            # Extract the Attention module
            residual_block = layer[0]
            prenorm = residual_block.fn
            attention_module = prenorm.fn
            
            # Forward pass through the layer
            x = layer(x)
            
            # Get attention scores and update rollout
            if hasattr(attention_module, 'last_attn'):
                # last_attn shape: (batch, seq_len, seq_len)
                attn = attention_module.last_attn  # (batch, seq_len, seq_len)
                
                # Rollout: accumulate attention across layers
                rollout = rollout @ (attn + I) # Product of attention maps with identity matrix
                rollout = rollout / rollout.sum(dim=-1, keepdim=True) # Normalize the product to get the rollout scores
                
                # Rollout: accumulate attention across layers
                # This multiplies the current attention matrix with the accumulated rollout
                # rollout = torch.matmul(attn, rollout)
                
        # Get attention from cls token (position 0) to patches (positions 1:)
        attn_scores = rollout[:, 0, 1:]  # cls token -> all patches
        return attn_scores
class ViT1DDecoder(nn.Module):
    def __init__(self, patch_size=16, hidden_dim=512, n_layers=3, heads=8, dropout=0.2565):
        super().__init__() 
        # Decoder hyperparameters
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.heads = heads
        self.dropout = dropout

        # Positional embedding
        self.positional_embedding = PositionalEncoding(hidden_dim= self.hidden_dim)

        # Decoder parameters
        self.decoder_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                Residual(PreNorm(hidden_dim, Attention(hidden_dim, n_heads=heads, dropout=dropout))),
                Residual(PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=dropout)))
            )
            for _ in range(n_layers)
        ])
        self.decoder_output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, patch_size),
            nn.Flatten(start_dim=1),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        # Using Xavier uniform initialization for linear layers
        for layer in self.decoder_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.xavier_uniform_(sublayer.weight)
                    if sublayer.bias is not None:
                        nn.init.zeros_(sublayer.bias)
    
    def forward(self, x):
        """Decodes the encoded tokens using the decoder."""
        # Add positional embedding
        x = self.positional_embedding(x)
        # Decoder projection
        x = self.decoder_projection(x)
        # Decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        # Remove cls token
        x = x[:, 1:]  
        # Decoder output
        decoded_tokens = self.decoder_output_head(x)

        return decoded_tokens