import importlib
import inspect
import torch
import numpy as np
import torch
from tqdm import tqdm
import os

from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split

# Mapping dataset names to their root directories
DATASET_PATHS = {
    "CWRU": "data/raw/CWRU",
    "LASPI": "data/raw/LASPI",
    "CVRTEST": "data/raw/CVR_Test",
}

def get_dataset(args:object):
    """
    Returns an instance of the specified dataset class.
    Args:
        name (str): Name of the dataset (e.g., 'CWRU', 'LASPI').
        **kwargs: Additional keyword arguments to pass to the dataset constructor.
    """
    name = args.name
    base_path = DATASET_PATHS[name.upper()]

    module = importlib.import_module(f"dataset.{name.lower()}")
    dataset_class = getattr(module, f"{name.upper()}Dataset")

    # Convert args -> dict
    args_dict = vars(args)
    
    # Inspecte la signature du constructeur
    sig = inspect.signature(dataset_class.__init__)
    valid_args = {
        k: v for k, v in args_dict.items()
        if k in sig.parameters and k != "self"
    }
    
    return dataset_class(root_dir=base_path, **valid_args)

def get_heterogeneous_split_dataloaders(args:object):
    name = args.name
    batch_size = args.batch_size

    # Get dataset
    dataset = get_dataset(args)

    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        # fallback: use a default collate_fn if not present
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # Split train/valid/test
    train_size = int(0.6 * len(dataset))
    valid_size = int(0.2 * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    # Generate split data
    generator =torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, 
                                                                            [train_size, valid_size, test_size], 
                                                                            generator=generator)
    
    # Split Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader

def make_strata(dataset):
    strata = []
    for sample in dataset:
        cls = sample['y_label']
        speed = sample['metadata']['speed']
        strata.append(f'{cls}_{speed}')
    strata =np.array(strata)
    return strata

def get_homogenous_split_dataloaders(args:object,seed:int=42):
    """
    Docstring for get_homogenous_split_dataloaders
    
    :param args: Description
    :type args: object
    :param seed: Description
    :type seed: int
    """ 
    name = args.name
    batch_size = args.batch_size
    data_ratio = args.data_ratio
    
    # Get dataset
    dataset = get_dataset(args)

    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        # fallback: use a default collate_fn if not present
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate

    # Create train, valid, test splits with stratification
    indice = np.arange(len(dataset))
    strata = make_strata(dataset)
    
    train_idx, test_val_idx = train_test_split(
        indice,
        test_size = 0.4,
        stratify=strata,
        random_state=42
    )
    
    valid_idx, test_idx = train_test_split(
        test_val_idx,
        test_size = 0.5,
        stratify=strata[test_val_idx],
        random_state=42
    )

    # réduire la taille du jeu de validation pour LASPI* pour éviter l'overfitting sur le jeu de validation
    if name == "LASPI":
        _, valid_idx = train_test_split(
            valid_idx,
            test_size=0.1, # Garder 10% des données de validation
            stratify=strata[np.isin(indice, valid_idx)],
            random_state=42
        )

    # Boucle sur les pourcentages de données étiquetées
    # Pas de random state pour avoir une variabilité entre les seeds
    if data_ratio < 1.0:
        _, scarcity_train_idx = train_test_split(
        train_idx,
        test_size = data_ratio,
        stratify=strata[train_idx],
        random_state=seed,
        )
        
    else :
        scarcity_train_idx = train_idx
    
    # Subdatasets stratified
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_idx)
    
    # Labels ponderation
    labels = [sample['y_label'] for sample in tqdm(train_dataset, desc="Collecting labels")]
    classes, class_counts = np.unique(labels, return_counts=True)
    
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in tqdm(zip(classes, class_weights), desc="Calculating class weights")}
    
    sample_weights = [class_weights[label] for label in tqdm(labels, desc="Calculating sample weights")]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, valid_loader, test_loader, np.unique(labels), dataset

def get_speed_stratified_dataloaders(args:object, train_val_speeds:list, test_speeds:list, seed:int=42):
    """
    Get train/valid and test dataloaders with different speed distributions.
    
    Args:
        args: Arguments with name, batch_size, data_ratio
        train_val_speeds: List of speeds to use for train and validation (e.g., [0, 1, 2])
        test_speeds: List of speeds to use for test (e.g., [3])
        seed: Random seed
    
    Returns:
        train_loader, valid_loader, test_loader, labels
    
    Example:
        # Use speeds 0, 1, 2 for training and speed 3 for testing
        train_loader, valid_loader, test_loader, labels = get_speed_stratified_dataloaders(
            args,
            train_val_speeds=[0, 1, 2],
            test_speeds=[3],
        )
    """
    name = args.name
    batch_size = args.batch_size
    data_ratio = args.data_ratio
    
    # Get full dataset
    dataset = get_dataset(args)
    
    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    
    # Get speed for each sample
    speeds = np.array([sample['metadata']['speed'] for sample in dataset])
    
    # Filter indices by speed
    train_val_mask = np.isin(speeds, train_val_speeds)
    test_mask = np.isin(speeds, test_speeds)
    
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # Create strata for train/val split
    strata_train_val = np.array([
        f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}"
        for i in tqdm(train_val_indices, desc="Creating strata for train/val split")
    ])
    
    # Split train/val with stratification
    train_idx, valid_idx = train_test_split(
        train_val_indices,
        test_size=0.25,  # 75% train, 25% valid
        stratify=strata_train_val,
        random_state=42
    )

    if name == "LASPI":
        # réduire la taille du jeu de validation pour LASPI*
        _, valid_idx = train_test_split(
            valid_idx,
            test_size=0.1, # Garder 10% des données de validation
            stratify=strata_train_val[np.isin(train_val_indices, valid_idx)],
            random_state=42
        )
    
    # Apply data scarcity to training set if needed
    if data_ratio < 1.0:
        strata_train = np.array([
            f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}"
            for i in tqdm(train_idx, desc="Creating strata for training set")
        ])
        scarcity_train_idx, _ = train_test_split(
            train_idx,
            train_size=data_ratio,
            stratify=strata_train,
            random_state=seed,
        )
    else:
        scarcity_train_idx = train_idx
    
    # Create subsets
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_indices)
    
    # Balanced sampling for training
    labels = [sample['y_label'] for sample in tqdm(train_dataset, desc="Collecting labels for balanced sampling")]
    classes, class_counts = np.unique(labels, return_counts=True)
    
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    sample_weights = [class_weights[label] for label in tqdm(labels, desc="Calculating sample weights")]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    
    return train_loader, valid_loader, test_loader, np.unique(labels), dataset

def get_speed_load_stratified_dataloaders(args:object, train_val_combinations:list, test_combinations:list, seed:int=42):
    """
    Get train/valid and test dataloaders with different speed and load combinations.
    
    Args:
        args: Arguments with name, batch_size, data_ratio
        train_val_combinations: List of (speed, load) tuples for train and validation (e.g., [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        test_combinations: List of (speed, load) tuples for test (e.g., [(3, 0), (3, 1)])
        seed: Random seed
    
    Returns:
        train_loader, valid_loader, test_loader, labels
    
    Example:
        # Use (speed, load) combinations (0,0), (0,1), (1,0), (1,1), (2,0), (2,1) for training
        # and (3,0), (3,1) for testing
        train_loader, valid_loader, test_loader, labels = get_speed_load_stratified_dataloaders(
            args,
            train_val_combinations=[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
            test_combinations=[(3, 0), (3, 1)],
        )
    """
    name = args.name
    batch_size = args.batch_size
    data_ratio = args.data_ratio
    
    # Get full dataset
    dataset = get_dataset(args)
    
    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    
    # Get speed and load for each sample
    speeds = np.array([sample['metadata']['speed'] for sample in dataset])
    loads = np.array([sample['metadata']['load'] for sample in dataset])
    
    # Filter indices by speed AND load combinations
    train_val_mask = np.zeros(len(dataset), dtype=bool)
    test_mask = np.zeros(len(dataset), dtype=bool)
    
    for speed, load in tqdm(train_val_combinations, desc="Filtering train/val indices"):
        train_val_mask |= (speeds == speed) & (loads == load)
    
    for speed, load in tqdm(test_combinations, desc="Filtering test indices"):
        test_mask |= (speeds == speed) & (loads == load)
    
    train_val_indices = np.where(train_val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    # Create strata for train/val split (include both speed and load)
    strata_train_val = np.array([
        f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}_{dataset[i]['metadata']['load']}"
        for i in tqdm(train_val_indices, desc="Creating strata for train/val split")
    ])
    
    # Split train/val with stratification
    train_idx, valid_idx = train_test_split(
        train_val_indices,
        test_size=0.25,  # 75% train, 25% valid
        stratify=strata_train_val,
        random_state=42
    )

    if name == "LASPI":
        # réduire la taille du jeu de validation pour LASPI*
        _, valid_idx = train_test_split(
            valid_idx,
            test_size=0.1, # Garder 10% des données de validation
            stratify=strata_train_val[np.isin(train_val_indices, valid_idx)],
            random_state=42
        )
    
    # Apply data scarcity to training set if needed
    if data_ratio < 1.0:
        strata_train = np.array([
            f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}_{dataset[i]['metadata']['load']}"
            for i in train_idx
        ])
        scarcity_train_idx, _ = train_test_split(
            train_idx,
            train_size=data_ratio,
            stratify=strata_train,
            random_state=seed,
        )
    else:
        scarcity_train_idx = train_idx
    
    # Create subsets
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_indices)
    
    # Balanced sampling for training
    labels = [sample['y_label'] for sample in train_dataset]
    classes, class_counts = np.unique(labels, return_counts=True)
    
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,num_workers=5, pin_memory=True)
    
    return train_loader, valid_loader, test_loader, np.unique(labels), dataset


def get_laspi_acquisition_split_dataloaders(args:object, seed:int=42):
    """
    Get train/valid and test dataloaders for LASPI dataset by splitting on acquisition files.
    
    For LASPI, each (fault_type, speed, load) condition has 3-4 acquisition files.
    This function uses files 1-2 for training/validation and file 3 for testing.
    
    Args:
        args: Arguments with name, batch_size, data_ratio
        seed: Random seed
    
    Returns:
        train_loader, valid_loader, test_loader, labels
    
    Example:
        train_loader, valid_loader, test_loader, labels = get_laspi_acquisition_split_dataloaders(args)
    """
    name = args.name
    batch_size = args.batch_size
    data_ratio = args.data_ratio
    
    # Get full dataset
    dataset = get_dataset(args)
    
    collate_fn = getattr(dataset, '_collate_fn', None)
    if collate_fn is None:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    
    # Extract acquisition file number from each sample's filepath
    def get_acquisition_number(filepath):
        """Extract acquisition number from filepath (e.g., 'acc_00001.csv' -> 1)"""
        basename = os.path.basename(filepath)
        # Format is typically 'acc_XXXXX.csv'
        if basename.startswith('acc_'):
            num_str = basename.replace('acc_', '').replace('.csv', '')
            try:
                return int(num_str)
            except ValueError:
                return -1
        return -1
    
    # Split indices based on acquisition file number
    train_val_indices = []
    test_indices = []
    
    for idx, sample in tqdm(enumerate(dataset), desc="Splitting LASPI by acquisition file"):
        acq_num = get_acquisition_number(sample['filepath'])
        
        # Files 1-3 for train/val, file 4+ for test
        if acq_num in [1, 2, 3]:
            train_val_indices.append(idx)
        elif acq_num >= 4:
            test_indices.append(idx)
    
    train_val_indices = np.array(train_val_indices)
    test_indices = np.array(test_indices)
    
    # Create strata for train/val split
    strata_train_val = np.array([
        f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}_{dataset[i]['metadata']['load']}"
        for i in tqdm(train_val_indices, desc="Creating strata for train/val split")
    ])
    
    # Split train/val with stratification
    train_idx, valid_idx = train_test_split(
        train_val_indices,
        test_size=0.10,  # 90% train, 10% valid
        stratify=strata_train_val,
        random_state=42
    )

    # Reduce shape of validation set for LASPI to avoid overfitting to validation set
    _, valid_idx = train_test_split(
        valid_idx,
        test_size=0.1, # Keep only 10% of validation data
        stratify=strata_train_val[np.isin(train_val_indices, valid_idx)],
        random_state=42
    )

    # Apply data scarcity to training set if needed
    if data_ratio < 1.0:
        strata_train = np.array([
            f"{dataset[i]['y_label']}_{dataset[i]['metadata']['speed']}_{dataset[i]['metadata']['load']}"
            for i in tqdm(train_idx, desc="Creating strata for training scarcity split")
        ])
        scarcity_train_idx, _ = train_test_split(
            train_idx,
            train_size=data_ratio,
            stratify=strata_train,
            random_state=seed,
        )
    else:
        scarcity_train_idx = train_idx
    
    # Create subsets
    train_dataset = Subset(dataset, scarcity_train_idx)
    valid_dataset = Subset(dataset, valid_idx)
    test_dataset = Subset(dataset, test_indices)
    
    # Balanced sampling for training
    labels = [sample['y_label'] for sample in tqdm(train_dataset, desc="Extracting labels for balanced sampling")]
    classes, class_counts = np.unique(labels, return_counts=True)
    
    class_weights = 1. / class_counts
    class_weights = class_weights / class_weights.sum()
    class_weights = {cls: weight for cls, weight in zip(classes, class_weights)}
    
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn, num_workers=5, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=5, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=5, pin_memory=True)
    
    return train_loader, valid_loader, test_loader, np.unique(labels), dataset

@torch.no_grad()
def compute_stats_from_dataloader(dataloader, device=None, verbose=True):
    """
    Calcule min et max de X_raw à partir d'un DataLoader.

    Args:
        dataloader: torch.utils.data.DataLoader
        device: torch.device ou None
        verbose: affiche une barre de progression

    Returns:
        dict {"min": float, "max": float}
    """
    min_val = float("inf")
    max_val = float("-inf")
    mean_val = 0.0
    std_val = 0.0

    iterator = dataloader
    if verbose:
        iterator = tqdm(iterator, desc="Computing stats from DataLoader")

    for batch in iterator:
        # X_raw = batch["X_raw"]
        X_raw = torch.log1p(batch["X_raw"])

        if device is not None:
            X_raw = X_raw.to(device, non_blocking=True)

        batch_min = X_raw.min().item()
        batch_max = X_raw.max().item()

        min_val = min(min_val, batch_min)
        max_val = max(max_val, batch_max)

        mean_val += X_raw.mean().item() * X_raw.size(0)
        std_val += X_raw.std().item() * X_raw.size(0)

    mean_val /= len(dataloader.dataset)
    std_val /= len(dataloader.dataset)

    return {"min": min_val, "max": max_val, "mean": mean_val, "std": std_val}