import os
import re
import torch
import numpy as np
from dataset.base import BaseDataset, Sample

class PHM07Dataset(BaseDataset):    
    def __init__(self, root_dir=None, 
                 fault_filter=None, 
                 speed_filter=None, 
                 window_size=None, 
                 window_stride=None, 
                 scaling_method=None,
                 downsampling_factor=None, 
                 symetric_spectrum=True,
                 use_float16=None):
        """
        Args:
            source (str): Source des données, peut être 'input' ou 'output' de la gearbox.
            fault_filter (list, optional): Liste de défauts à filtrer. Si None, tous les défauts sont inclus.
            transform_type (str, optional): Type de transformation à appliquer aux données.
            window_size (int, optional): Taille de la fenêtre pour les échantillons.
            stride (int, optional): Pas de la fenêtre pour les échantillons.
        """
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        assert speed_filter is None or isinstance(speed_filter, list), "speed_filter doit être une liste ou None"

        # Définir les attributs de l'instance
        self.root_dir = root_dir
        self.fault_filter = fault_filter
        self.speed_filter = speed_filter
        
        # Définir les types de roulements et les conditions de charge        
        self.sampling_frequency = 2e5 # en Hz

        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter,
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         scaling_method=scaling_method,
                         downsampling_factor=downsampling_factor,
                         symetric_spectrum=symetric_spectrum,
                         use_float16=use_float16,
                         )
    
    def __delattr__(self, name):
        """
        Permet de supprimer un attribut de l'instance.
        """
        return super().__delattr__(name)

    def _read_sample(self, filepath)-> torch.Tensor:
        """
        Lit un sample à partir du fichier .csv spécifié.
        Args:
            filepath (str): Chemin vers le fichier .csv
        Returns:
            torch.Tensor: Un tenseur contenant les données lues.
        """
        txt_data = np.loadtxt(filepath)  # Lit les données CSV en ignorant la première ligne (en-tête)

        data = torch.tensor(txt_data[:,0], dtype=torch.float32)
        return data


    def _collect_samples(self):
        """
        Collecte les échantillons depuis les fichiers .csv dans le répertoire PHM09.
        Pour cette base de données, il n'y a pas de labels.
        """
        for file in os.listdir(self.root_dir):
            if not file.endswith(".md"):
                filepath = os.path.join(self.root_dir, file)
        
                self.samples.append(Sample(
                    filepath=filepath,
                    label=None,
                    metadata={}
                ))
                            
