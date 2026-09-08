import os
import torch
import numpy as np
import pandas as pd
from dataset.base import BaseDataset, Sample

class LASPIDataset(BaseDataset):
    def __init__(self, root_dir=None, fault_filter=None, speed_filter=None, window_size=2048, window_stride=256, downsampling_factor=None, symetric_spectrum=True):
        """
        Args:
            transform (callable, optional): Transformation à appliquer aux données.
        """
        super().__init__(root_dir=root_dir, 
                         fault_filter=fault_filter, 
                         speed_filter=speed_filter,
                         window_size=window_size, 
                         window_stride=window_stride,
                         downsampling_factor=downsampling_factor,
                         symetric_spectrum=symetric_spectrum, # LASPI dataset has non symetric spectrum
                         )
        
        # Fréquences d'échantillonnage pour chaque condition de fonctionnement
        self.sampling_frequencies = 25600  # En Hz, à ajuster selon les conditions spécifiques
        
        # Description des arbres
        self.shafts = ['shaft_1','shaft_2','shaft_3']

        # Description des engrenages (nombre de dents)
        self.gears_descriptions = {
            "Z1_tooth_number": 29,
            "Z2a_tooth_number": 100,
            "Z2b_tooth_number": 36,
            "Z3_tooth_number": 20,
        }

        # Ratios de réduction entre les engrenages
        self.reduction_ratios = {
            "R1/1_reduction_ratio": self.gears_descriptions["Z1_tooth_number"] / self.gears_descriptions["Z1_tooth_number"],
            "R2/1_reduction_ratio": self.gears_descriptions["Z1_tooth_number"] / self.gears_descriptions["Z2a_tooth_number"],
            "R3/1_reduction_ratio": self.gears_descriptions["Z2b_tooth_number"] / self.gears_descriptions["Z3_tooth_number"] * self.gears_descriptions["Z1_tooth_number"] / self.gears_descriptions["Z2a_tooth_number"],
        }
        
        # Description des roulements (dimensions et caractéristiques)
        self.bearing_descriptions = {
            "ball_number": 9,
            "ball_diameter_mm": 7.938,
            "pitch_diameter_mm": 38.499,
            "contact_angle_deg": 0,
        }
        
    
    def __delattr__(self, name):
        """
        Permet de supprimer un attribut de l'instance.
        """
        return super().__delattr__(name)

    def _read_sample(self, filepath)-> torch.Tensor:
        """
        Lit un sample à partir du fichier CSV spécifié.
        Args:
            filepath (str): Chemin vers le fichier CSV.
        Returns:
            Sample: Un objet Sample contenant les données lues.
        """
        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        data = np.transpose(data)  # Transpose pour avoir les colonnes comme caractéristiques
        data = data[3] # On récupère les données de la quatrième colonne (accéléromètre)
        data = torch.tensor(data, dtype=torch.float32)
        return data

    def _collect_samples(self):
        """
        Docstring for _collect_samples
        
        :param self: Description
        """ 
        # Parcours des répertoires pour collecter les échantillons
        for default in os.listdir(self.root_dir):
            default_path = os.path.join(self.root_dir, default)
            if not os.path.isdir(default_path) or default == '__pycache__':
                continue

            # Parcours des conditions de fonctionnement
            for conditions in os.listdir(default_path):
                speed, load, speed_rpm = conditions.split('_')
                
                # Nettoyage des valeurs
                speed = int(speed.replace('hz', ''))
                load = int(load.replace('%', ''))
                speed_rpm = int(speed_rpm.replace('rpm', ''))

                # Chemin vers le répertoire des conditions
                cond_path = os.path.join(default_path, conditions)
                if not os.path.isdir(cond_path):
                    continue

                # Parcours des fichiers CSV dans le répertoire des conditions
                for file in os.listdir(cond_path):
                    if file.endswith('.csv'):
                        csv_path = os.path.join(cond_path, file)
                        label = self._extract_label_from_filename(default) # Convertit le nom du défaut en étiquette
                        
                        # self.fault_filter = ['inner','outer','normal']

                        # Filtrage des échantillons selon les filtres spécifiés
                        if self.fault_filter is  None or label in self.fault_filter:
                            if self.speed_filter is None or speed in self.speed_filter:

                                # Ajout de l'échantillon à la liste des échantillons
                                self.samples.append(Sample(filepath=csv_path,
                                                            label=label,
                                                            metadata={'speed': speed, 
                                                                      'load': load, 
                                                                      'speed_rpm': speed_rpm}))
    def _extract_label_from_filename(self, default):

        mapping = {'Bearing_inner_race_fault': 'inner', 
                   'Bearing_outer_race_fault': 'outer', 
                   'Gear_half_broken_tooth': 'gear_half', 
                   'Gear_half_broken_tooth_and_bearing_outer_race_faults': 'gear_half_&_outer',
                   'Gear_surface_and_bearing_inner_race_faults': 'gear_surface_&_inner',
                   'Gear_surface_damage': 'gear_surface',
                   'Healthy_motor': 'normal',
                   }

        return mapping.get(default, -1)

    def extract_caracteristic_freq(self, batch):
        """
        Docstring for extract_caracteristic_freq
        :param self: Description
        :param batch: Description
        :return: Description
        """ 
        # Extraction des fréquences caractéristiques pour chaque échantillon du batch
        metadata = pd.DataFrame(batch['metadata'])
        speed_rpm = metadata['speed_rpm'].values/60  # Convertir les tours par minute en tours par seconde

        # Calcul des fréquences de rotation et des fréquences de défaut pour chaque arbre
        shaft_bearing_freqs = dict()
        mesh_gear_freqs = dict()

        for shaft in self.shafts:

            reduction_ratio = self.reduction_ratios[f"R{shaft.split('_')[1]}/1_reduction_ratio"]
            shaft_rotation_frequency = speed_rpm*reduction_ratio

            # Bearing frequencies calculation
            n = self.bearing_descriptions['ball_number']
            d = self.bearing_descriptions['ball_diameter_mm']
            D = self.bearing_descriptions['pitch_diameter_mm']
            alpha = np.radians(self.bearing_descriptions['contact_angle_deg'])

            shaft_bearing_freqs[shaft] = {"f_r": shaft_rotation_frequency,  # Fréquence de rotation
                                    'f_BPFO': n/2 * shaft_rotation_frequency * (1 - d/D * np.cos(alpha)),  # Fréquence de passage de défaut sur la piste extérieure
                                    'f_BPFI': n/2 * shaft_rotation_frequency * (1 + d/D * np.cos(alpha)),  # Fréquence de passage de défaut sur la piste intérieure
                                    'f_FTF': shaft_rotation_frequency/2 * (1 - d/D * np.cos(alpha)),  # Fréquence de passage de défaut sur les rouleaux
                                    'f_BSF': shaft_rotation_frequency * D/d * (1 - (d/D * np.cos(alpha))**2)  # Fréquence de passage de défaut sur les rouleaux
                                    }
            
            # Convertir les fréquences en DataFrame pour une meilleure lisibilité
            shaft_bearing_freqs[shaft] = pd.DataFrame(shaft_bearing_freqs[shaft])
    

        # computing gear mesh frequencies
        for i in range(len(self.shafts)-1):
            shaft_input = self.shafts[i]
            shaft_output = self.shafts[i+1]

            shaft_rotation_frequency = shaft_bearing_freqs[shaft_input]['f_r']

            if shaft_input == 'shaft_1' and shaft_output == 'shaft_2':
                gear_tooth_number = self.gears_descriptions["Z1_tooth_number"]

            elif shaft_input == 'shaft_2' and shaft_output == 'shaft_3':
                gear_tooth_number = self.gears_descriptions[f"Z2b_tooth_number"]

            mesh_gear_freqs[f"f_GMF_{shaft_input}_to_{shaft_output}"] = shaft_rotation_frequency * gear_tooth_number
            mesh_gear_freqs[f"f_GMF_{shaft_input}_to_{shaft_output}"] = pd.DataFrame(mesh_gear_freqs[f"f_GMF_{shaft_input}_to_{shaft_output}"])
        
        return shaft_bearing_freqs, mesh_gear_freqs
    
    def defauts_freqs(self, batch):
        """
        Docstring for defauts_freqs
        :param self: Description
        :param batch: Description
        :return: Description
        """ 
        shaft_bearing_freqs, mesh_gear_freqs = self.extract_caracteristic_freq(batch)

        df_bearing = shaft_bearing_freqs['shaft_2']
        df_gear = mesh_gear_freqs['f_GMF_shaft_1_to_shaft_2']
        df_gear.columns = ['f_GMF']

        # Combinaison des fréquences de défauts dans un seul DataFrame
        df_defauts = pd.concat([df_bearing, df_gear], axis=1)
        return df_defauts
        
    def create_expert_mask(self, dataloader, patch_size:int):
        """
        Crée un masque d'attention expert au niveau des patches.
        Chaque patch reçoit un score basé sur la présence de fréquences de défaut.
        
        :param self: Description
        :param batch: Description (contenant X_raw de shape (batch_size, spectrum_size))
        :param patch_size: Taille de chaque patch en fréquences
        :return: mask de shape (batch_size, number_patch) avec scores normalisés
        """         
        all_masks = []

        for batch in dataloader:
            df_defauts = self.defauts_freqs(batch)
            batch_size = batch['X_raw'].shape[0]  # Nombre d'échantillons dans le batch
            spectrum_size = batch['X_raw'].shape[1]  # Taille du spectre (ex: 2048)
            number_patch = spectrum_size // patch_size  # Nombre de patches

            # Création du masque par patch (batch_size, number_patch)
            mask = np.zeros((batch_size, number_patch), dtype=np.float32)

            # Parcours de chaque échantillon du batch
            for sample_idx, (_, row) in enumerate(df_defauts.iterrows()):
                # Pour chaque fréquence de défaut de cet échantillon
                for freq in row:
                    if isinstance(freq, (int, float)) and freq < spectrum_size:
                        # Déterminer à quel patch appartient cette fréquence
                        patch_idx = int(freq / patch_size)
                        # S'assurer que l'indice est valide
                        if 0 <= patch_idx < number_patch:
                            mask[sample_idx, patch_idx] += 1.0  # Incrémenter le score du patch

            # Normaliser le masque (optionnel, pour avoir des valeurs soit de 0 ou 1)
            mask[np.where(mask > 0)] = 1.0  # Binariser le masque (présence ou absence de fréquences de défaut)

            # max_val = mask.max()
            # if max_val > 0:
            #     mask = mask / max_val
            
            all_masks.append(mask)
        
        all_masks = np.concatenate(all_masks, axis=0)  # Combiner les masques de tous les batches

        return all_masks

