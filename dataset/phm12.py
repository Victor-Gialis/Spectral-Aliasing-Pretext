import os
import re
import torch
import numpy as np
from dataset.base import BaseDataset, Sample

class PHM12Dataset(BaseDataset):   
    # On a prit seulement le Test set de PHM12 pour le moment, car base de données est grande
    def __init__(self, root_dir=None, 
                 source='input',
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
        assert source in ['input', 'output'], "source doit être 'input' ou 'output'"
        assert fault_filter is None or isinstance(fault_filter, list), "fault_filter doit être une liste ou None"
        assert speed_filter is None or isinstance(speed_filter, list), "speed_filter doit être une liste ou None"

        # Définir les attributs de l'instance
        self.root_dir = root_dir
        self.source = source
        self.fault_filter = fault_filter
        self.speed_filter = speed_filter
        
        # Définir les types de roulements et les conditions de charge        
        self.sampling_frequency = 25.6e3 # en Hz

        # https://www.indatlas.com/datasets/

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
        try:
            csv_data = np.loadtxt(filepath, delimiter=',')  # Lit les données CSV en ignorant la première ligne (en-tête)
        except Exception as e:
            csv_data = np.loadtxt(filepath, delimiter=';')  # Lit les données CSV en ignorant la première ligne (en-tête)

        data = torch.tensor(csv_data[:, -1], dtype=torch.float32)
        return data


    def _collect_samples(self):
        """
        Collecte les échantillons depuis les fichiers .csv dans le répertoire PHM09.
        Pour cette base de données, il n'y a pas de labels.
        """
        for dir in os.listdir(self.root_dir):
            if os.path.isdir(os.path.join(self.root_dir, dir)):
                dir_path = os.path.join(self.root_dir, dir)

                match = re.search(r'Bearing(\d+)', dir)
                condition_number = int(match.group(1)) if match else None
                speed, load = self._extract_speed_load_from_condition(condition_number)

                for file in os.listdir(dir_path):
                    if file.endswith(".csv"):
                        filepath = os.path.join(dir_path, file)

                        try:
                            self.samples.append(Sample(
                                filepath=filepath,
                                label=None,
                                metadata={
                                    'speed': speed,
                                    'load': load,
                                }
                            ))
                        except Exception as e:
                            print(f"Erreur lors du traitement du fichier {filepath}: {e}")


    def _extract_speed_load_from_condition(self, condition_number):
        """
        Extrait la vitesse et la charge à partir du numéro de condition.
        Args:
            condition_number (int): Numéro de la condition (1, 2 ou 3)
        Returns:
            tuple: (vitesse, charge) en RPM et N respectivement"""
        speed = None # en RPM
        load = None # en N

        if condition_number == 1:
            speed = 1800
            load = 4000

        elif condition_number == 2:
            speed = 1650
            load = 4200

        elif condition_number == 3:
            speed = 1500
            load = 5000

        return speed, load
