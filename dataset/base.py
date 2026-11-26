import os
import torch
import numpy as np

from tqdm import tqdm
from dataclasses import dataclass
from scipy.signal import hilbert
from torch.utils.data import Dataset
from torch import stack, Tensor
from dataset.transform import spectrum, undersampling

@dataclass
class Sample:
    filepath: str
    label: str
    metadata: dict

@dataclass
class SampleWindow:
    filepath: str
    label: str = None
    data : torch.tensor = None
    start_idx: int = 0
    metadata: dict = None

class BaseDataset(Dataset):
    def __init__(self, root_dir = None, 
                 fault_filter=None, 
                 speed_filter=None, 
                 window_size=None, 
                 window_stride=None, 
                 scaling_method=None,
                 downsampling_factor=None,
                 symetric_spectrum=True,
                 ):
        """
        Args:
            root_dir (str): Répertoire racine contenant les données.
            fault_filter (list, optional): Liste des défauts à inclure. Si None, tous les défauts sont inclus.
            speed_filter (list, optional): Liste des vitesses à inclure. Si None, toutes les vitesses sont incluses.
            window_size (int, optional): Taille de la fenêtre pour découper les samples. Si None, pas de découpage.
            window_stride (int, optional): Pas de la fenêtre pour découper les samples. Si None, pas de découpage.
            downsampling_factor (int, optional): Facteur de sous échantillonnage. Si None, pas de sous échantillonnage.
            symetric_spectrum (bool, optional): Indique si le spectre doit être symétrique. Par défaut True.
        """
        assert os.path.isdir(root_dir), f"Le répertoire {root_dir} n'existe pas ou n'est pas un répertoire."
        assert window_size is None or isinstance(window_size, int) and window_size > 0, "window_size doit être un entier positif."
        assert window_stride is None or isinstance(window_stride, int) and window_stride > 0, "window_stride doit être un entier positif."
        assert scaling_method in [None, 'normalization', 'standardization'], "scaling_method doit être None, 'normalization' ou 'standardization'."

        self.root_dir = root_dir
        self.fault_filter = fault_filter
        self.speed_filter = speed_filter
        self.window_size = window_size
        self.window_stride = window_stride
        self.scaling_method = scaling_method
        self.downsampling_factor = downsampling_factor # Downsampling factor
        self.symetric_spectrum = symetric_spectrum

        self.samples = []
        self.windows = []

        self.stats = dict(zip(['time','frequency'],[{"min": 0, "max": 0, "mean": 0, "std": 0} for k in range(2)]))

        self._collect_samples()
        self._collect_windows()
    
    def _read_sample(self, filepath):
        """
        Lit un sample à partir du fichier spécifié.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("Chaque dataset doit implémenter _read_sample()")

    def _collect_samples(self):
        raise NotImplementedError("Chaque dataset doit implémenter _collect_samples()")

    def _extract_label_from_filename(self, filename):
        """
        Convertit le nom du défaut en étiquette.
        Doit être implémenté par les sous-classes.
        """
        raise NotImplementedError("Chaque dataset doit implémenter _label_from_filename()")

    def _collect_windows(self):
        """
        Collecte les fenêtres de samples en fonction de window_size et window_stride.
        Si window_size et window_stride ne sont pas définis, chaque sample est considéré comme une fenêtre unique.
        """

        ##### FOR THE TIME DOMAIN STATS #####
        min_val_time = float("inf")
        max_val_time = float("-inf")

        n_time = 0
        sum_x_time = 0
        sum_x2_time = 0

        ##### FOR THE FREQUENCY DOMAIN STATS #####
        min_val_freq = float("inf")
        max_val_freq = float("-inf")

        n_freq = 0
        sum_x_freq = 0
        sum_x2_freq = 0
    
        for sample in tqdm(self.samples, desc="Collecting windows"):

            try:
                data = self._read_sample(sample.filepath)
                signal_length = data.shape[-1] # Longueur du signal, supposé être le dernier axe

                if self.window_size is None or self.window_stride is None:
                    self.windows.append(SampleWindow(filepath=sample.filepath, label=sample.label, data=data, start_idx=0, metadata=sample.metadata))
                else:
                    for start in range(0, signal_length - self.window_size + 1, self.window_stride):
                        self.windows.append(SampleWindow(filepath=sample.filepath, label=sample.label, data=data[..., start:start + self.window_size],start_idx=start, metadata=sample.metadata))

            except Exception as e:
                print(f"✗ Failed to read sample {sample.filepath}: {str(e)}")
                continue

            # Get the raw magnitude spectrum of the last window for stats computation
            x_raw = self.windows[-1].data # time domain
            X_raw = spectrum.reduced_magnitude_spectrum(x_raw) # frequency domain

            ##### FOR THE TIME DOMAIN STATS #####
            batch_min = x_raw.min().item()
            batch_max = x_raw.max().item()

            min_val_time = min(min_val_time, batch_min)
            max_val_time = max(max_val_time, batch_max)

            n_time += X_raw.numel()
            sum_x_time += X_raw.sum().item()
            sum_x2_time += (X_raw ** 2).sum().item()

            ##### FOR THE FREQUENCY DOMAIN STATS #####
            batch_min = X_raw.min().item()
            batch_max = X_raw.max().item()

            min_val_freq = min(min_val_freq, batch_min)
            max_val_freq = max(max_val_freq, batch_max)

            n_freq += X_raw.numel()
            sum_x_freq += X_raw.sum().item()
            sum_x2_freq += (X_raw ** 2).sum().item()

        # Compute mean and std for all domains
        ##### FOR THE TIME DOMAIN STATS #####
        self.stats["time"]["mean"] = sum_x_time / n_time
        var = (sum_x2_time / n_time) - (self.stats["time"]["mean"] ** 2)
        self.stats["time"]["std"] = var ** 0.5
        self.stats["time"]["min"] = min_val_time
        self.stats["time"]["max"] = max_val_time

        ##### FOR THE FREQUENCY DOMAIN STATS #####  
        self.stats["frequency"]["mean"] = sum_x_freq / n_freq
        var = (sum_x2_freq / n_freq) - (self.stats["frequency"]["mean"] ** 2)
        self.stats["frequency"]["std"] = var ** 0.5
        self.stats["frequency"]["min"] = min_val_freq
        self.stats["frequency"]["max"] = max_val_freq

    def _collate_fn(self, batch):
        batch_out = {}
        keys = batch[0].keys()
        for key in keys:
            values = [item[key] for item in batch]
            # Si tous les éléments sont des tensors de même taille, on stack
            if all(isinstance(v, Tensor) for v in values):
                try:
                    batch_out[key] = stack(values)
                except Exception:
                    batch_out[key] = values  # fallback: liste si stack impossible
            else:
                batch_out[key] = values
        return batch_out
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        Args:
            idx (int): index to retrieve
        Returns:
            dict: A dictionary containing:
                - 'X_raw': Raw magnitude spectrum of the original time series.
                - 'X_folded': Processed magnitude spectrum of the downsampled time series.
                - 'y_label': Label associated with the sample.
                - 'metadata': Metadata associated with the sample.
        """
        # Reading the sample
        filepath = self.windows[idx].filepath
        y_label = self.windows[idx].label
        metadata = self.windows[idx].metadata
        
        # Time series
        x_raw = self.windows[idx].data
        # x_raw = x_raw.to(torch.float16) # Convert to float16 for processing
        x_fold = None

        if x_raw is None:
            raise ValueError(f"The sample at index {idx} could not be read.")

        # Frequency domain transformation
        X_raw = spectrum.reduced_magnitude_spectrum(x_raw)

        # Downsampling is None = random factor for each sample
        if self.downsampling_factor is None or self.downsampling_factor == "None":
            self.downsampling_factor = np.random.randint(1, 10) # Random downsampling factor between 1 and 10

        # Downsampling is int = fixed factor for all samples
        elif type(self.downsampling_factor) == str :
            self.downsampling_factor = int(self.downsampling_factor)

        x_fold = undersampling.undersampling_time_serie(x_raw, factor=self.downsampling_factor)
        X_fold = spectrum.reduced_magnitude_spectrum(x_fold)

        if self.symetric_spectrum:
            X_fold = spectrum.symetric_spectrum_preparation(X_fold, factor=self.downsampling_factor, target_size=X_raw.shape[-1])

        # Time domain
        x_raw_scaled = self._normalize_time_tensor(x_raw) if self.scaling_method == 'normalize' else self._standardize_time_tensor(x_raw)
        x_fold_scaled = self._normalize_time_tensor(x_fold) if self.scaling_method == 'normalize' else self._standardize_time_tensor(x_fold)

        # Frequency domain
        X_raw_scaled = self._normalize_frequency_tensor(X_raw) if self.scaling_method == 'normalize' else self._standardize_frequency_tensor(X_raw)
        X_fold_scaled = self._normalize_frequency_tensor(X_fold) if self.scaling_method == 'normalize' else self._standardize_frequency_tensor(X_fold)

        sample_rate = (metadata or {}).get('sampling_frequency')
        if sample_rate is None:
            sample_rate = getattr(self, 'sampling_frequency', getattr(self, 'sampling_frequencies', 0))

        return {# Time domain
                'x_raw':x_raw,                                                                                                                                      # Raw time series
                'x_raw_scaled':x_raw_scaled,                  # Standardized time series
                'x_folded':x_fold,                                                                                                                                  # Downsampled time series
                'x_folded_scaled':x_fold_scaled,             # Standardized downsampled time series
                
                # Frequency domain
                'X_raw':X_raw,                                                                                                                                      # Magnitude spectrum of the original time series
                'X_raw_scaled':X_raw_scaled,        # Standardized magnitude spectrum of the original time series
                'X_folded':X_fold,                                                                                                                                  # Magnitude spectrum of the downsampled time series
                'X_folded_scaled':X_fold_scaled,   # Standardized magnitude spectrum of the downsampled time series
                
                # Metadata
                'y_label':y_label,                                              # Label associated with the sample
                'sample_rate_hz': torch.tensor(sample_rate, dtype=torch.float32),
                'sampling_stride': torch.tensor(self.downsampling_factor, dtype=torch.long),
                'metadata':metadata,                                            # Metadata associated with the sample
                'stats':self.stats,                                             # Dataset statistics
                'index':idx,                                                    # Index of the sample
                'filepath': filepath}                                           # Filepath of the sample

    def _normalize_time_tensor(self, x_raw):
        """
        Normalizes the time domain tensor using the stored min and max values.
        Args:
            x_raw (torch.Tensor): The time domain tensor to be normalized.
        Returns:
            torch.Tensor: The normalized time domain tensor.
        """
        min_val = self.stats['time']['min']
        max_val = self.stats['time']['max']

        return (x_raw - min_val) / (max_val - min_val)  # Min-max normalization


    def _normalize_frequency_tensor(self, X_freq):
        """
        Normalizes the frequency domain tensor using the stored min and max values.
        Args:
            X_freq (torch.Tensor): The frequency domain tensor to be normalized.
        Returns:
            torch.Tensor: The normalized frequency domain tensor.
        """
        min_val = self.stats['frequency']['min']
        max_val = self.stats['frequency']['max']
        
        return (X_freq - min_val) / (max_val - min_val)  # Min-max normalization

    def _standardize_time_tensor(self, x_raw):
        """
        Standardizes the time domain tensor using the stored mean and std values.
        Args:
            x_raw (torch.Tensor): The time domain tensor to be standardized.
        Returns:
            torch.Tensor: The standardized time domain tensor.
        """
        mean_val = self.stats['time']['mean']
        std_val = self.stats['time']['std']

        return (x_raw - mean_val) / std_val  # Z-score standardization

    def _standardize_frequency_tensor(self, X_freq):
        """
        Standardizes the frequency domain tensor using the stored mean and std values.
        Args:
            X_freq (torch.Tensor): The frequency domain tensor to be standardized.
        Returns:
            torch.Tensor: The standardized frequency domain tensor.
        """
        mean_val = self.stats['frequency']['mean']
        std_val = self.stats['frequency']['std']

        return (X_freq - mean_val) / std_val  # Z-score standardization