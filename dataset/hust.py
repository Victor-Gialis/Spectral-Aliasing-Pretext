import os
import re
import torch
import numpy as np
from scipy.io import loadmat
from dataset.base import BaseDataset, Sample

class HUSTDataset(BaseDataset):    
    def __init__(self, root_dir=None, 
                 bearing_type=None,
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
        
        # Définir les types de roulements et les conditions de charge
        self.bearing_types = [None, None, None, None, 6204, 6205, 6206, 6207, 6208]  # Index 0-3 non utilisés
        self.load_conditions = [0, None, 200, None, 400]  # Correspondance: 0→0W, 2→200W, 4→400W
        
        self.sampling_frequency = 51200 # en Hz
        self.sampling_duration = 10 # en secondes

        # https://www.indatlas.com/datasets/

        # Exemple : I402.mat --> I :Inner defect / 4:6204 type / 2:200W condition working

        # 6204 bearing type (mm):
        # - inner diameter : 20
        # - outer diameter : 47
        # - ball diameter : 7.6
        # - ball number : 8

        # 6205 bearing type (mm)):
        # - inner diameter : 25
        # - outer diameter : 52
        # - ball diameter : 7.8
        # - ball number : 9

        # 6206 bearing type (mm):
        # - inner diameter : 30
        # - outer diameter : 62
        # - ball diameter : 9.0
        # - ball number : 9

        # 6207 bearing type (mm):
        # - inner diameter : 35
        # - outer diameter : 72
        # - ball diameter : 11.0
        # - ball number : 9

        # 6208 bearing type (mm):
        # - inner diameter : 40
        # - outer diameter : 80
        # - ball diameter : 12.0
        # - ball number : 9


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
        Lit un sample à partir du fichier .mat spécifié.
        Args:
            filepath (str): Chemin vers le fichier .mat
        Returns:
            torch.Tensor: Un tenseur contenant les données lues.
        """
        
        mat_data = loadmat(filepath)
        # Récupère les données associées à self.source (ex: 'I402')
        data = torch.tensor(mat_data['data' ]).float().squeeze()

        return data

    def _collect_samples(self):

        for mat_path in os.listdir(self.root_dir):
            if mat_path.endswith(".mat"):

                # On suppose que le nom de fichier est toujours au format I402.mat
                filename = mat_path.split('.')[0]
                match = re.match(r"([A-Za-z]{1,2})(\d{3})$", filename)

                if match:
                    letters = match.group(1)
                    numbers = match.group(2)

                else :
                    raise ValueError(f"Le nom de fichier {filename} ne correspond pas au format attendu.")

                # On extrait le type de roulement et la condition de fonctionnement à partir du nom du fichier
                label= self._extract_label_from_filename(letters)
                bearing_type = self.bearing_types[int(numbers[0])]
                load = self.load_conditions[int(numbers[-1])]

                if self.bearing_type is None or bearing_type in self.bearing_type:
                    # On ne garde que les fichiers qui correspondent à la source
                    if self.fault_filter is None or label in self.fault_filter :
                        # On ne garde que les fichiers qui correspondent au filtre de défaut et à la source
                        if self.speed_filter is None or load in self.speed_filter:
                        # On ne garde que les fichiers qui correspondent au filtre de vitesse
                            filepath = os.path.join(self.root_dir, mat_path)
                            speed = self._extract_speed_from_filedata(filepath)
                            self.samples.append(Sample(filepath=filepath,
                                                label=label,
                                                metadata={'speed': speed, 
                                                          'load': load,
                                                        'bearing_type': bearing_type,
                                                }))
                            
    def _extract_speed_from_filedata(self, filepath):
        mat_data = loadmat(filepath)
        speed = mat_data['fs'][0][0] * 60 # Convertit la fréquence d'échantillonnage en tours par minute (RPM)

        return int(speed)

    def _extract_label_from_filename(self, default):
        # labels de classes standardisés
        mapping = {'N':'normal',
                    'I':'inner', 
                    'O':'outer', 
                    'B':'ball',
                    'IB':'inner_&_ball',
                    'IO':'inner_&_outer',
                    'OB':'outer_&_ball',}
        
        return mapping.get(default, -1)

    def _extract_bearing_type_from_filename(self, numbers):
        # On suppose que le premier chiffre du numéro correspond au type de roulement
        mapping = {
            '4': 6204,
            '5': 6205,      
            '6': 6206,
            '7': 6207,
            '8': 6208,}
        
        return mapping.get(numbers, -1)

    def _extract_working_condition_from_filename(self, numbers):
        # On suppose que le dernier chiffre du numéro correspond à la condition de fonctionnement
        mapping = {
            '0': 0,
            '2': 200,      
            '4': 400,}
        
        return mapping.get(numbers, -1)

