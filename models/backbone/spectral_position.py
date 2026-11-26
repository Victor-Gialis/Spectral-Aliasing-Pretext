import torch
from torch import nn


def spectral_patch_coordinates(batch, patch_size, *, folded):
    """Return each patch's observed center frequency and bandwidth in Hz."""
    signal = batch["x_folded" if folded else "x_raw"]
    spectrum = batch["X_folded" if folded else "X_raw"]
    factor = batch["sampling_stride"] if folded else torch.ones_like(batch["sampling_stride"])
    sample_rate = batch["sample_rate_hz"]
    if not torch.all(torch.isfinite(sample_rate) & (sample_rate > 0)):
        raise ValueError('Frequency positional encoding requires positive sample rates in Hz')
    spacing = sample_rate / (factor * signal.shape[-1])

    # The spectrum transform drops the final rFFT bin.
    observed_bins = signal.shape[-1] // 2
    indices = torch.arange(spectrum.shape[-1], device=spectrum.device)
    copy = indices // observed_bins
    local_index = indices % observed_bins
    if folded:
        valid = copy[None, :] < factor[:, None]
        indices = torch.where(copy % 2 == 0, local_index, observed_bins - 1 - local_index)
    else:
        valid = torch.ones_like(indices, dtype=torch.bool).expand(spectrum.shape[0], -1)

    indices = indices.expand(spectrum.shape[0], -1).unfold(-1, patch_size, patch_size)
    valid = valid.unfold(-1, patch_size, patch_size)
    count = valid.sum(dim=-1)
    center = (indices * valid).sum(dim=-1) / count.clamp_min(1)
    first = indices.masked_fill(~valid, observed_bins).amin(dim=-1)
    last = indices.masked_fill(~valid, -1).amax(dim=-1)
    width = (last - first + 1).clamp_min(0)
    coordinates = torch.stack((center, width), dim=-1) * spacing[:, None, None]
    return coordinates, count > 0


class SpectralPositionalEncoding(nn.Module):
    """Continuous positions with the same physical scale for every dataset."""

    def __init__(self, hidden_dim):
        super().__init__()
        # Frequencies are expressed in kHz before applying Fourier features.
        self.register_buffer("scales", torch.logspace(-2, 1, steps=8), persistent=False)
        self.projection = nn.Linear(32, hidden_dim)

    def forward(self, tokens, positions):
        if positions is None:
            raise ValueError("Frequency positional encoding requires spectral patch coordinates")
        coordinates, valid = positions
        angles = coordinates.float().unsqueeze(-1) * (2 * torch.pi * self.scales / 1000)
        features = torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(start_dim=-2)
        encoding = self.projection(features.to(self.projection.weight.dtype))
        encoding = encoding.to(tokens.dtype).masked_fill(~valid.unsqueeze(-1), 0)
        return torch.cat((tokens[:, :1], tokens[:, 1:] + encoding), dim=1)
