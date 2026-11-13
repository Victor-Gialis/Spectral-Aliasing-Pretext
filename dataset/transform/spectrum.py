import torch

def reduced_magnitude_spectrum(x):
    """
    Compute the reduced magnitude spectrum of a time series tensor.
    Args:
        x (torch.Tensor): numpy array or of length N representing the time series data.
    Returns:
        torch.Tensor: Tensor containing the reduced magnitude spectrum.
    """
    # psd = torch.abs(torch.fft.rfft(data - torch.mean(data)))/N

    x -= torch.mean(x, dim=-1, keepdim=True)  # Center the signal
    n = len(x)

    # Compute the Fast Fourier Transform (FFT)
    fft_result = torch.fft.rfft(x)
    
    # Compute the magnitude spectrum
    reduced_spectrum = torch.abs(fft_result)/n

    return torch.log1p(reduced_spectrum[:-1]) # Exclude the Nyquist frequency component

def symetric_spectrum_preparation(spectrum, factor=1, target_size=None):
    """
    Prepare a symmetric spectrum by concatenating the original and flipped version of the spectrum.
    Args:
        spectrum (torch.Tensor): Input tensor representing the spectrum data.
        factor (int): Downsampling factor used. Number of repetitions of spectrum pattern.
    Returns:
        torch.Tensor: Tensor containing the symmetric spectrum prepared for folding.

    """
    complete_spectrum = [spectrum]

    for i in range(1, factor):
        if i % 2 == 1:
            # Odd iterations: add flipped spectrum
            complete_spectrum.append(torch.flip(spectrum, [-1]))
        else:
            # Even iterations: add original spectrum
            complete_spectrum.append(spectrum)
    
    complete_spectrum = torch.cat(complete_spectrum, dim=-1)
    
    if target_size is not None:
        if complete_spectrum.shape[-1] < target_size:
            # Padding with zeros if the complete spectrum is smaller than the target size
            padding_size = target_size - complete_spectrum.shape[-1]
            complete_spectrum = torch.nn.functional.pad(complete_spectrum, (0, padding_size))
            
        else:
            complete_spectrum = complete_spectrum[..., :target_size]

    return complete_spectrum