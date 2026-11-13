from dataset import dataloader

def split_dataloader(split_type=None,args_dataloader=None):
    """
    This function is use to split dataset_name before training, validation and testing. It takes as input the type of split to perform and the dataloader arguments, and returns the corresponding dataloaders for training, validation and testing.
    Args:
        split_type: Type of split to perform. Can be one of "independent", "speed_stratified", "speed_load_stratified" or "sample_stratified".
    Returns :
        train_loader: Dataloader for training
        valid_loader: Dataloader for validation
        test_loader: Dataloader for testing
        labels: The list of class labels in the dataset (useful for downstream classification tasks)
        dataset: The full dataset object (useful for computing stats for normalization)
    """
    assert split_type in ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"] , f"Invalid split type: {split_type}"
    
    # Get dataset_name name from args_dataloader
    dataset_name = args_dataloader.name if hasattr(args_dataloader, 'name') else KeyError("dataset_name name not specified") # Default to LASPI if not specified

    # Get seed if exists, otherwise default to 0
    seed = args_dataloader.seed if hasattr(args_dataloader, 'seed') else 0

    # Split dataloaders - sampling spectrum independently
    if split_type == "independent":
        train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_homogenous_split_dataloaders(args_dataloader,seed=seed)

    # Split dataloaders - speed stratified sampling
    elif split_type == "speed_stratified":

        if dataset_name == "CWRU":
            # Speeds: 1730,1750, 1772 and 1797 RPM for CWRU
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_speed_stratified_dataloaders(
                args=args_dataloader,
                train_val_speeds=['1750', '1772', '1797'],
                test_speeds=['1730'],
                seed=seed,
            )
        
        elif dataset_name == "LASPI":
            # Speeds: 25, 35 and 45 Hz for LASPI
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_speed_stratified_dataloaders(
                args=args_dataloader,
                train_val_speeds=[35, 45],
                test_speeds=[25],
                seed=seed,
            )

        elif dataset_name == "PADERBORN":
            # Speeds: 1000, 1500 and 2000 RPM for PADERBORN
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_speed_stratified_dataloaders(
                args=args_dataloader,
                train_val_speeds=[1500],
                test_speeds=[900],
                seed=seed,
            )

        else:
            raise NotImplementedError(f"Speed stratified sampling not implemented for {dataset_name} yet")

    # Split dataloaders - speed-load stratified sampling
    elif split_type == "speed_load_stratified":

        if dataset_name == "LASPI":
            # Loads : 0, 25, 50 and 75 % of full load for LASPI
            # Speeds : 25, 35 and 45 Hz for LASPI
            # Total combinations : 12 (all combinations of 4 loads and 3 speeds) - test on (25Hz, 50% load)
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_speed_load_stratified_dataloaders(
                args=args_dataloader, 
                train_val_combinations=[(25, 0), (25, 25), (25, 75),
                                        (35, 0), (35, 25), (35, 50), (35, 75),
                                        (45, 0), (45, 25), (45, 50), (45, 75),],
                test_combinations=[(25, 50)],
                seed=seed,
            )

        elif dataset_name == "PADERBORN":
            # Loads : 0, 25, 50 and 75 % of full load for PADERBORN
            # Speeds : 1000, 1500 and 2000 RPM for PADERBORN
            # Total combinations : 12 (all combinations of 4 loads and 3 speeds) - test on (1000RPM, 50% load)
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_speed_load_stratified_dataloaders(
                args=args_dataloader, 
                train_val_combinations=[(1000, 0), (1000, 25), (1000, 75),
                                        (1500, 0), (1500, 25), (1500, 50), (1500, 75),
                                        (2000, 0), (2000, 25), (2000, 50), (2000, 75),],
                test_combinations=[(1000, 50)],
                seed=seed,
            )

        else:
            raise NotImplementedError(f"Speed-load stratified sampling not implemented for {dataset_name} yet")
    
    elif split_type == "sample_stratified":
        
        if dataset_name == "LASPI":
            # Mixed speed train/val, test on fixed speeds and load conditions, but only 1 load condition per speed to avoid overfitting to load conditions
            train_loader, valid_loader, test_loader, labels, dataset = dataloader.get_laspi_acquisition_split_dataloaders(
                args=args_dataloader,
                seed=seed,
            )

        else :
            raise NotImplementedError(f"Sample stratified sampling not implemented for {dataset_name} yet")

    return train_loader, valid_loader, test_loader, labels, dataset
