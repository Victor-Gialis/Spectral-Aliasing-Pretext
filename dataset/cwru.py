import os
import torch
import numpy as np
from dataset.base import BaseDataset, Sample

class CWRUDataset(BaseDataset):    
    def __init__(self, root_dir=None, source='FE', fault_filter=None, speed_filter=None, window_size=2048, window_stride=256, downsampling_factor=None, symetric_spectrum=True):
        """
        Args:
            source (str): Source des données, peut être 'DE', 'FE' ou 'BA'.
            fault_filter (list, optional): Liste de défauts à filtrer. Si None, tous les défauts sont inclus.
            transform_type (str, optional): Type de transformation à appliquer aux données.
            window_size (int, optional): Taille de la fenêtre pour les échantillons.
            stride (int, optional): Pas de la fenêtre pour les échantillons.
        """
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        assert speed_filter is None or isinstance(speed_filter, list), "speed_filter doit être une liste ou None"
        
        self.source = 'FE'

        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter,
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         downsampling_factor=downsampling_factor,
                         symetric_spectrum=symetric_spectrum,
                         )
    
    def __delattr__(self, name):
        """
        Permet de supprimer un attribut de l'instance.
        """
        return super().__delattr__(name)

    def _read_sample(self, filepath)-> torch.Tensor:
        """
        Lit un sample à partir du fichier npz spécifié.
        Args:
            filepath (str): Chemin vers le fichier npz.
        Returns:
            Sample: Un objet Sample contenant les données lues.
        """
        data = np.load(filepath)
        data = torch.tensor(data[self.source]).float().squeeze()
        return data

    def _collect_samples(self):

        for speed in os.listdir(self.root_dir):
            
            full_speed_path = os.path.join(self.root_dir, speed)
            if os.path.isdir(full_speed_path):
                for f in os.listdir(full_speed_path):
                    if f.endswith(".npz"):
                        npz_path = os.path.join(full_speed_path, f)
                        filename = f.split('.')[0]
                        pattern = filename.split('_')
                        # On suppose que le nom de fichier est toujours au format speed_fault_diameter_end
                        # Par exemple : 1700_1_1_DE.npz ou 1700_1_1_1_DE.npz
                        diameter = 'none'
                        position = 'none'

                        if len(pattern) == 4:
                            # Cas des fichiers avec position et diamètre du défaut
                            speed, default, diameter, source = pattern
                            if '@' in default:
                                default, position = default.split('@')
                        else:
                            # Cas des fichiers en fonctionnement normal
                            speed, default = pattern
                            source = self.source  # Pour les fichiers en acquisition normale, on suppose que c'est la fin par défaut
                        
                        # On extrait le label standardisé du nom du fichier
                        label = self._extract_label_from_filename(default)

                        if self.source is None or source in self.source:
                            # On ne garde que les fichiers qui correspondent à la source
                            if self.fault_filter is None or label in self.fault_filter and self.source in source:
                                # On ne garde que les fichiers qui correspondent au filtre de défaut et à la source
                                if self.speed_filter is None or speed in self.speed_filter:
                                    # On ne garde que les fichiers qui correspondent au filtre de vitesse
                                    label = label + '_' + diameter 
                                    self.samples.append(Sample(filepath=npz_path,
                                                        label=label,
                                                        metadata={'speed': speed, 
                                                                'position': position, 
                                                                'diameter': diameter, 
                                                                'source': source}))

    def _extract_label_from_filename(self, default):
        # labels de classes standardisés
        mapping = {'Normal':'normal',
                    'IR':'inner', 
                    'OR':'outer', 
                    'B':'ball'}

        return mapping.get(default, -1)

