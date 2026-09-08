# MIT License

# Copyright (c) 2022 Zhihan Yue

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from types import SimpleNamespace

class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding,
            dilation=dilation,
            groups=groups
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0
        
    def forward(self, x):
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, : -self.remove]
        return out  # (batch_size, seq_length, output_dim)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class DilatedConv(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels
        self.kernel_size = kernel_size

        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)

### Use in the TS2Vec model as the backbone encoder to extract features from time series data. The encoder consists of a series of dilated convolutional blocks that capture temporal dependencies at multiple scales.
class DilatedCNNEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, depth=5, kernel_size=3, dropout=0.1, domain="time", normalization=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.kernel_size = kernel_size
        self.dropout = dropout  # Dropout rate for regularization
        self.domain = domain
        self.normalization = normalization

        # Pretrain dataset Stats
        self.stats = dict()

        # Input projection layer to map input dimensions to hidden dimensions
        self.input_projector = nn.Linear(input_dim, hidden_dim//2)

        # Define the channels for the dilated convolutional encoder
        channels = [hidden_dim//2] * depth + [hidden_dim] # Final layer outputs the desired output dimensions
        self.encoder = nn.Sequential(
            DilatedConv(in_channels=hidden_dim//2,
                            channels=channels, 
                            kernel_size=kernel_size),
            nn.Dropout(p=self.dropout)  # Apply dropout for regularization,
        )

        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Apply input projection
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add channel dimension if input is (batch_size, seq_length)

        x = self.input_projector(x)# Project to hidden_dim//2
        x = x.transpose(1, 2)  # (batch_size, hidden_dim//2, seq_length) for Conv1d

        # Pass through the dilated convolutional encoder
        embeded_x = self.encoder(x)

        return embeded_x.transpose(1, 2)  # (batch_size, seq_length, output_dim)
    
    def _initialize_weights(self):
        # Initialize weights if necessary
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _loads_stats(self, stats):
        self.stats = stats

    def _get_arguments(self):
        """
        Returns:
            args (SimpleNamespace): Arguments of the model 
        """
        return SimpleNamespace(
            input_dim = self.input_dim,
            hidden_dim = self.hidden_dim,
            depth = self.depth,
            kernel_size = self.kernel_size
        )


### Use in TS-TCC architecture as the backbone encoder to extract features from time series data. The encoder consists of a series of dilated convolutional blocks that capture temporal dependencies at multiple scales.
class CNNEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, kernel_size=3, stride=1, dropout=0.1, domain="time", normalization=None):
        super().__init__()

        # Encoder hyperparameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout  # Dropout rate for regularization
        self.domain = domain
        self.normalization = normalization

        # Pretrain dataset Stats
        self.stats = dict()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=32, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      padding=kernel_size//2,
                      bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=dropout),

            nn.Conv1d(in_channels=32, 
                      out_channels=64, 
                      kernel_size=8, 
                      stride=1, 
                      padding=4,
                      bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=dropout),

            nn.Conv1d(in_channels=64, 
                      out_channels=hidden_dim, 
                      kernel_size=8, 
                      stride=1, 
                      padding=4,
                      bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=dropout),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # Add channel dimension if input is (batch_size, seq_length)
        # x: (batch_size, seq_length, channels)
        
        x = x.transpose(1, 2)  # (batch_size, channels, seq_length) for Conv1d
        embeded_x = self.conv_block(x)

        return embeded_x.transpose(1, 2) # (batch_size, seq_length, hidden_dim)

    def _initialize_weights(self):
        # Initialize weights if necessary
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _loads_stats(self, stats):
        self.stats = stats
    
    def _get_arguments(self):
        """
        Returns:
            args (SimpleNamespace): Arguments of the model 
        """
        return SimpleNamespace(
            input_dim = self.input_dim,
            hidden_dim = self.hidden_dim,
            kernel_size = self.kernel_size,
            stride = self.stride
        )



    


