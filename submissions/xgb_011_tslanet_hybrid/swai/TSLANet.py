import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft

import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    Positional Encoding for time series patches
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(1)]
        return x

class PatchEmbedding(nn.Module):
    """
    Split time series into patches and project to embedding dimension
    """
    def __init__(self, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        batch_size, channels, seq_len = x.shape

        # Pad sequence if needed
        if seq_len % self.patch_size != 0:
            pad_len = self.patch_size - (seq_len % self.patch_size)
            x = torch.cat([x, torch.zeros(batch_size, channels, pad_len, device=x.device)], dim=-1)
            seq_len += pad_len

        # Split into patches
        num_patches = seq_len // self.patch_size
        x = x.view(batch_size, channels, num_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_patches, channels, patch_size]
        x = x.reshape(batch_size, num_patches, -1)  # Flatten channels and patch_size

        # Project to embedding dimension
        x = self.proj(x)
        return x

class AdaptiveSpectralBlock(nn.Module):
    """
    Adaptive Spectral Block (ASB) that performs frequency domain processing
    with adaptive noise filtering
    """
    def __init__(self, embed_dim, threshold_quantile=0.9):
        super().__init__()
        self.embed_dim = embed_dim
        self.threshold_quantile = threshold_quantile

        # Learnable global and local filters
        self.global_filter = nn.Parameter(torch.randn(embed_dim, dtype=torch.cfloat))
        self.local_filter = nn.Parameter(torch.randn(embed_dim, dtype=torch.cfloat))

        # Learnable threshold parameter
        self.threshold = nn.Parameter(torch.tensor(0.5))

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def adaptive_high_freq_mask(self, x_fft):
        """
        Create adaptive mask for high frequency components
        """
        # Calculate power spectrum
        power = torch.abs(x_fft).pow(2)

        # Compute adaptive threshold
        threshold = torch.quantile(power, self.threshold_quantile)

        # Create mask (1 for frequencies to keep, 0 for those to filter)
        mask = (power > threshold * self.threshold).float()
        return mask

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, embed_dim = x.shape

        # Apply FFT along the sequence dimension
        x_fft = fft.fft(x, dim=1)

        # Adaptive filtering
        mask = self.adaptive_high_freq_mask(x_fft)
        x_filtered = x_fft * mask

        # Apply global and local filters
        x_global = x_fft * self.global_filter
        x_local = x_filtered * self.local_filter

        # Combine and inverse FFT
        x_combined = x_global + x_local
        x_out = fft.ifft(x_combined, dim=1).real

        # Layer normalization
        x_out = self.norm(x_out)
        return x_out

class InteractiveConvolutionBlock(nn.Module):
    """
    Interactive Convolution Block (ICB) with parallel convolutions
    """
    def __init__(self, embed_dim, kernel_sizes=[3, 5]):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_sizes[0], padding='same')
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_sizes[1], padding='same')
        self.conv3 = nn.Conv1d(embed_dim, embed_dim, 1)  # Final mixing convolution
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embed_dim]
        """
        # Permute for convolution
        x_perm = x.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]

        # First convolution path
        conv1_out = self.conv1(x_perm)
        conv2_out = self.conv2(x_perm)

        # Interactive multiplication
        a1 = self.activation(conv1_out) * conv2_out
        a2 = self.activation(conv2_out) * conv1_out

        # Combine and final convolution
        combined = a1 + a2
        output = self.conv3(combined)

        # Permute back and normalize
        output = output.permute(0, 2, 1)
        output = self.norm(output + x)  # Residual connection
        return output

class TSLANetLayer(nn.Module):
    """
    A single TSLANet layer composed of ASB and ICB
    """
    def __init__(self, embed_dim, kernel_sizes=[3, 5]):
        super().__init__()
        self.asb = AdaptiveSpectralBlock(embed_dim)
        self.icb = InteractiveConvolutionBlock(embed_dim, kernel_sizes)

    def forward(self, x):
        x = self.asb(x)
        x = self.icb(x)
        return x

class TSLANet(nn.Module):
    """
    Complete TSLANet model
    """
    def __init__(self, in_channels, patch_size, embed_dim, num_layers,
                 num_classes=None, forecast_horizon=None, kernel_sizes=[3, 5]):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # TSLANet layers
        self.layers = nn.ModuleList([
            TSLANetLayer(embed_dim, kernel_sizes) for _ in range(num_layers)
        ])

        # Output heads
        self.num_classes = num_classes
        self.forecast_horizon = forecast_horizon

        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)
        if forecast_horizon is not None:
            self.forecaster = nn.Linear(embed_dim, forecast_horizon)

    def forward(self, x):
        # x: [batch_size, channels, seq_len]
        
        # Patch embedding
        x = self.patch_embed(x)
       
        # Positional encoding
        x = self.pos_encoder(x)

        # TSLANet layers
        for layer in self.layers:
            x = layer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # [batch_size, embed_dim]

        # Output heads
        outputs = {}
        if self.num_classes is not None:
            outputs['classification'] = self.classifier(x)
        if self.forecast_horizon is not None:
            outputs['forecasting'] = self.forecaster(x)

        return outputs
