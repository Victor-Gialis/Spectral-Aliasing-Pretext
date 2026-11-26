import os
import torch
import numpy as np
from tqdm import tqdm
from dataset.base import BaseDataset, Sample

class CVRTESTDataset(BaseDataset):
    def __init__(self, 
                 root_dir=None, 
                 fault_filter=None, 
                 speed_filter=None, 
                 window_size=2048, 
                 window_stride=256, 
                 downsampling_factor=None, 
                 scaling_method=None):
        """
        Dataset pour CVR_Test avec données temporelles multivariées.
        
        Args:
            root_dir (str): Répertoire racine contenant les données (input.pt et output.pt).
            fault_filter (list, optional): Liste de défauts à filtrer. Si None, tous les défauts sont inclus.
            speed_filter (list, optional): Non utilisé pour CVR_Test.
            window_size (int, optional): Taille de la fenêtre pour les échantillons.
            window_stride (int, optional): Pas de la fenêtre pour les échantillons.
            downsampling_factor (int, optional): Facteur de sous-échantillonnage.
        """
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        
        # Mapping des labels CVR_Test
        self.label_mapping = {
            0: 'nominal',
            1: 'overvoltage',
            2: 'undervoltage',
            3: 'right_offset',
            4: 'left_offset',
            5: 'nominal'
        }
        
        # Fréquence d'échantillonnage
        self.sampling_frequency = 25600  # 25.6 kHz
        
        # Cache pour les données (chargées une seule fois)
        self.inputs_cache = None
        
        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter,
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         scaling_method=scaling_method,
                         downsampling_factor=downsampling_factor
                         )
    
    def __delattr__(self, name):
        """
        Permet de supprimer un attribut de l'instance.
        """
        return super().__delattr__(name)

    def _collect_samples(self):
        """
        Collecte les échantillons de CVR_Test.
        Charge les fichiers input.pt et output.pt une seule fois et les stocke en cache.
        """
        input_path = os.path.join(self.root_dir, 'input.pt')
        output_path = os.path.join(self.root_dir, 'output.pt')
        
        # Vérifier que les fichiers existent
        if not os.path.exists(input_path) or not os.path.exists(output_path):
            raise FileNotFoundError(f"Les fichiers input.pt et output.pt doivent être présents dans {self.root_dir}")
        
        # Charger les données une seule fois et les stocker en cache
        inputs = torch.load(input_path)  # Shape: (1380, 5, 25600)
        outputs = torch.load(output_path)  # Shape: (1380,)
        
        # Sélection un des 5 capteurs (par exemple, le capteur 0)
        inputs = inputs[:,:,-2]  # Shape: (1380, 25600)

        # S'assurer que les données sont des tensors
        if isinstance(inputs, np.ndarray):
            inputs = torch.tensor(inputs, dtype=torch.float32)
            print(f"✓ Inputs convertis en torch.Tensor: shape {inputs.shape}, dtype {inputs.dtype}")
        if isinstance(outputs, np.ndarray):
            outputs = torch.tensor(outputs, dtype=torch.long)
            print(f"✓ Outputs convertis en torch.Tensor: shape {outputs.shape}, dtype {outputs.dtype}")
        
        # Stocker les données en cache pour la lecture ultérieure
        self.inputs_cache = inputs
        
        # Créer un sample pour chaque acquisition
        for idx in tqdm(range(len(inputs)), desc="Collecting samples"):
            label_id = int(outputs[idx].item())
            label = self._extract_label_from_filename(label_id)
            
            # Filtrage selon les critères
            if self.fault_filter is None or label in self.fault_filter:
                # Créer un chemin virtuel pour identifier le sample
                virtual_path = f"cvr_test_{idx}"
                
                self.samples.append(Sample(filepath=virtual_path,
                                          label=label,
                                          metadata={'index': idx, 
                                                   'label_id': label_id,
                                                   'speed': 3, #En m/s, la vitesse du tapis
                                                   'sampling_frequency': self.sampling_frequency}))

    def _read_sample(self, filepath) -> torch.Tensor:
        """
        Lit un sample à partir du chemin virtuel en utilisant le cache.
        Args:
            filepath (str): Chemin virtuel du format "cvr_test_{idx}".
        Returns:
            torch.Tensor: Les données du sample.
        """
        # Extraire l'index du chemin virtuel
        if isinstance(filepath, str) and filepath.startswith('cvr_test_'):
            idx = int(filepath.split('_')[-1])
            
            # Utiliser les données en cache
            if self.inputs_cache is not None:
                data = self.inputs_cache[idx]  # Shape: (25600,)
                return data
            else:
                raise RuntimeError("Le cache des inputs n'a pas été initialisé")
        else:
            raise ValueError(f"Chemin invalide: {filepath}")

    def _extract_label_from_filename(self, label_id):
        """
        Convertit un ID de label en étiquette textuelle.
        Args:
            label_id (int): ID du label (0-5).
        Returns:
            str: Étiquette textuelle.
        """
        return self.label_mapping.get(label_id, 'unknown')
