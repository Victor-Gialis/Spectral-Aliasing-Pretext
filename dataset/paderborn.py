import os
import re
import torch
import numpy as np
from scipy.io import loadmat
from dataset.base import BaseDataset, Sample

class PADERBORNDataset(BaseDataset):    
    def __init__(self, root_dir=None, 
                 bearing_type=None,
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
            source (str): Source des données, peut être 'DE', 'FE' ou 'BA'.
            fault_filter (list, optional): Liste de défauts à filtrer. Si None, tous les défauts sont inclus.
            transform_type (str, optional): Type de transformation à appliquer aux données.
            window_size (int, optional): Taille de la fenêtre pour les échantillons.
            stride (int, optional): Pas de la fenêtre pour les échantillons.
        """
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        assert speed_filter is None or isinstance(speed_filter, list), "speed_filter doit être une liste ou None"

        # Définir les attributs de l'instance
        self.root_dir = root_dir
        self.bearing_type = bearing_type
        self.fault_filter = fault_filter
        self.speed_filter = speed_filter
        
        self.sampling_frequency = 6.4e4 # en Hz

        # https://www.indatlas.com/datasets/

        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter,
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         scaling_method=scaling_method,
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
        Lit un sample à partir du fichier .mat spécifié.
        Args:
            filepath (str): Chemin vers le fichier .mat
        Returns:
            torch.Tensor: Un tenseur contenant les données lues.
        """
        filename = os.path.basename(filepath).split('.')[0]  # Extrait le nom du fichier sans l'extension
        mat_data = loadmat(filepath)
        # Récupère les données associées à self.source (ex: 'I402')
        # data = torch.tensor(mat_data['data' ]).float().squeeze()

        Y = mat_data[filename][0, 0]['Y']
        vibration = None

        for signal in Y[0]:
            name = signal['Name'].item()

            if name == "vibration_1":
                vibration = torch.tensor(signal['Data']).float().squeeze()
                break

        if vibration is None:
            raise ValueError(f"Le signal de vibration n'a pas été trouvé dans le fichier {filepath}")

        return vibration

    def _collect_samples(self):
        """
        Collecte les échantillons depuis les fichiers .mat du dataset Paderborn.
        Format Paderborn: Répertoires par défaut (Normal, Outer Race Fault, Inner Race Fault, Ball Fault)
        """
        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)):
                for mat_file in os.listdir(os.path.join(self.root_dir, dir)):
                    if mat_file.endswith('.mat'):
                        try:
                            filepath = os.path.join(self.root_dir, dir, mat_file)
                            filename = os.path.basename(filepath).split('.')[0]

                            matches = re.findall(r"([A-Za-z])(\d+)", filename)
                            values = {letter: int(number) for letter, number in matches}

                            speed = values.get('N', None)*100  # Extrait la vitesse (RPM)
                            torque = values.get('M', None)/10  # Extrait le couple (Nm)
                            load = values.get('F', None)*100  # Extrait la charge radiale (N)

                            label = self._extract_label_from_filename(mat_file)

                            # Filtrer par vitesse (RPM)
                            if self.speed_filter is None or speed in self.speed_filter:

                                self.samples.append(Sample(
                                    filepath=filepath,
                                    label=label,
                                    metadata={
                                        'speed': speed,
                                        'load': load,
                                        'torque': torque,
                                    }
                                ))

                        except Exception as e:
                            print(f"Erreur lors du traitement du fichier {filepath}: {e}")


    def _extract_label_from_filename(self, filename):
        # labels de classes standardisés pour Paderborn
        # Exemple filename: N09_M07_F10_K002_1.mat
        # We extract the fault type from the filename and map it to a standardized label
        default = filename.split('_')[3][:2]  # Extrait la partie du défaut (ex: 'K0', 'KA', 'KI')
        mapping = {
            'K0': 'normal',
            'KA': 'outer',
            'KI': 'inner',
        }
        
        return mapping.get(default, -1)


