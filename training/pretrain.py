import csv,os, json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from models.ssl.ijepa import IJEPAModel
from models.ssl.mae import MAEModel
from models.ssl.sap import SAPModel
from models.ssl.ts2vec import TS2VecModel
from models.ssl.tstcc import TSTCCModel

# Utilities for pretraining
def create_run_dir(method:str, dataset:str)->Path:
    """
    Create a directory to save training results.
    Args:
        method (str): Name of the pretraining method.
        dataset (str): Name of the dataset.
    Returns:
        run_dir (Path): Path to the created directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("results")/"pretrain"/f"{method}/{dataset}/{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def log_metrics(run_dir:Path, epoch:int, train_loss:float, valid_loss:float):
    """
    Log training and validation metrics to a CSV file.
    Args:
        run_dir (Path): Directory where metrics are logged.
        epoch (int): Current epoch number.
        train_loss (float): Training loss.
        valid_loss (float): Validation loss.
    """
    csv_path = run_dir / "log" / "metrics.csv"
    os.makedirs(csv_path.parent, exist_ok=True)
    file_exists = csv_path.exists() # Check if file already exists
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['epoch', 'train_loss', 'valid_loss'])
        writer.writerow([epoch, train_loss, valid_loss])

def plot_metrics(run_dir:Path):
    """
    Plot training and validation loss curves.
    Args:
        run_dir (Path): Directory where metrics are logged.
    """
    csv_path = run_dir / "log" / "metrics.csv"
    if not csv_path.exists():
        print("No metrics to plot.")
        return

    df = pd.read_csv(csv_path)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
    plt.plot(df['epoch'], df['valid_loss'], label='Valid Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(run_dir / "log" / "loss_plot.png")
    plt.close()

def save_model_config(run_dir:Path, 
                      args_dataloader:object, 
                      args_backbone:object, 
                      args_ssl:object, 
                      args_training:object):
    """
    Save model configuration in a json file
    Args:
       run_dir (Path): Directory where model configuration is saved.
       args_dataloader (object): Dataloader configuration
       args_backbone (object): Backbone configuration
       args_ssl (object): SSL configuration
       args_training (object): Training configuration
    """
    config = {
        "dataset":vars(args_dataloader).copy(),
        "backbone":vars(args_backbone).copy(),
        "ssl":vars(args_ssl).copy(),
        "training":vars(args_training).copy(),
    }

    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

def save_model_checkpoint(run_dir:Path, model:nn.Module, name:str="model.pt"):
    """
    Save model checkpoint.
    Args:
        run_dir (Path): Directory to save the checkpoint.
        model (nn.Module): The model to be saved.
        name (str): Name of the checkpoint file.
    """
    checkpoint_path = run_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    # Create checkpoint dictionary with model state and backbone metadata
    checkpoint = {
        'model_state': model.state_dict(),
    }
    
    # Save backbone metadata if it exists
    if hasattr(model, 'backbone'):
        # Save backbone arguments (hyperparameters)
        if hasattr(model.backbone, '_get_arguments'):
            checkpoint['backbone_args'] = vars(model.backbone._get_arguments())

        # Save backbone stats
        if hasattr(model.backbone, 'stats'):
            checkpoint['backbone_stats'] = model.backbone.stats

        # Save backbone domain if it exists
        if hasattr(model.backbone, 'domain'):
            checkpoint['backbone_domain'] = model.backbone.domain

        # Save backbone normalization if it exists
        if hasattr(model.backbone, 'normalization'):
            checkpoint['backbone_normalization'] = model.backbone.normalization
    
    # Save additional SSL parameters if MAE model
    if isinstance(model, MAEModel):
        if hasattr(model, 'mask_ratio'):
            checkpoint['mask_ratio'] = model.mask_ratio
    
    #  Save downsampling factor if it exists
    if isinstance(model, SAPModel):
        if hasattr(model, 'downsample_factor'):
            checkpoint['downsample_factor'] = model.downsample_factor

    # Save additional SSL parameters if IJEPA model
    if isinstance(model, IJEPAModel):
        if hasattr(model, 'enc_mask_ratio'):
            checkpoint['enc_mask_ratio'] = model.enc_mask_ratio
        if hasattr(model, 'pred_mask_ratio'):
            checkpoint['pred_mask_ratio'] = model.pred_mask_ratio
        if hasattr(model, 'momentum'):
            checkpoint['momentum'] = model.momentum
    
    # Save additional SSL parameters if TS2Vec model
    if isinstance(model, TS2VecModel):
        if hasattr(model, 'temperature'):
            checkpoint['temperature'] = model.temperature
        if hasattr(model, 'use_cosine_similarity'):
            checkpoint['use_cosine_similarity'] = model.use_cosine_similarity
        if hasattr(model, 'temporal_unit'):
            checkpoint['temporal_unit'] = model.temporal_unit

    # Save additional SSL parameters if TS-TCC model
    if isinstance(model, TSTCCModel):
        if hasattr(model, 'temperature'):
            checkpoint['temperature'] = model.temperature
        if hasattr(model, 'timesteps'):
            checkpoint['timesteps'] = model.timesteps
    
    torch.save(checkpoint, checkpoint_path / name)

def load_model_checkpoint(model:nn.Module, checkpoint_path:Path)->nn.Module:
    """
    Load model checkpoint and restore backbone stats and arguments.
    Args:
        model (nn.Module): The model to load weights into.
        checkpoint_path (Path): Path to the checkpoint file.
    Returns:
        nn.Module: The model with loaded weights and stats.
    """
    checkpoint_path = os.path.join(checkpoint_path,'checkpoints','last.pt') if isinstance(checkpoint_path, str) else checkpoint_path
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    # Load model state
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        # Load with strict=False to handle missing keys (e.g., mask_token added in newer versions)
        incompatible_keys = model.load_state_dict(checkpoint['model_state'], strict=False)
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in checkpoint: {incompatible_keys.missing_keys}")
        if incompatible_keys.unexpected_keys:
            print(f"Warning: Unexpected keys in checkpoint: {incompatible_keys.unexpected_keys}")
        
        # Restore backbone metadata if available
        if hasattr(model, 'backbone'):
            if 'backbone_stats' in checkpoint:
                model.backbone._loads_stats(checkpoint['backbone_stats'])
            
            if 'backbone_domain' in checkpoint:
                model.backbone.domain = checkpoint['backbone_domain']
            
            if 'backbone_normalization' in checkpoint:
                model.backbone.normalization = checkpoint['backbone_normalization']

            # Restore SSL method metadata if MAE model
            if isinstance(model, MAEModel):
                if 'mask_ratio' in checkpoint:
                    model.mask_ratio = checkpoint['mask_ratio']

            # Restore SSL method metadata if SAP model
            if isinstance(model, SAPModel):
                if 'downsample_factor' in checkpoint:
                    model.downsample_factor = checkpoint['downsample_factor']

            # Restore SSL method metadata if IJEPA model
            if isinstance(model, IJEPAModel):
                if 'enc_mask_ratio' in checkpoint:
                    model.enc_mask_ratio = checkpoint['enc_mask_ratio']
                if 'pred_mask_ratio' in checkpoint:
                    model.pred_mask_ratio = checkpoint['pred_mask_ratio']
                if 'momentum' in checkpoint:
                    model.momentum = checkpoint['momentum']
                    
            # Restore SSL method metadata if TS2Vec model
            elif isinstance(model, TS2VecModel):
                if 'temperature' in checkpoint:
                    model.temperature = checkpoint['temperature']
                if 'use_cosine_similarity' in checkpoint:
                    model.use_cosine_similarity = checkpoint['use_cosine_similarity']
                if 'temporal_unit' in checkpoint:
                    model.temporal_unit = checkpoint['temporal_unit']

            # Restore SSL method metadata if TS-TCC model
            elif isinstance(model, TSTCCModel):
                if 'temperature' in checkpoint:
                    model.temperature = checkpoint['temperature']
                if 'timesteps' in checkpoint:
                    model.timesteps = checkpoint['timesteps']
    else:
        # Backward compatibility: if checkpoint is just state_dict
        incompatible_keys = model.load_state_dict(checkpoint, strict=False)
        if incompatible_keys.missing_keys:
            print(f"Warning: Missing keys in checkpoint: {incompatible_keys.missing_keys}")
    
    return model

def move_batch_to_device(batch, device:torch.device):
    """
    Move batch data to the specified device.
    Args:
        batch (dict): Batch data containing tensors.
        device (torch.device): Target device.
    Returns:
        dict: Batch data on the target device.
    """
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
    return batch

def evaluate(run_dir:Path,
            device:torch.device, 
            model:nn.Module, 
            test_loader:torch.utils.data.DataLoader
            ):
    """
    Evaluation loop for self-supervised learning models.
    Args:
        run_dir (Path): Directory to save logs and visualizations.
        device (torch.device): Device to run the evaluation on.
        model (nn.Module): The model to be evaluated.
        test_loader (DataLoader): DataLoader for test data.
    """
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = move_batch_to_device(batch, device)
            outputs = model(inputs)
            
            loss = model.compute_loss(outputs, inputs)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)   
    print(f"Test Loss: {test_loss:.4f}")

    # Visulisation of random batch tensor
    for batch in test_loader:
        inputs = move_batch_to_device(batch, device)
        outputs = model(inputs)

        # Get raw and predicted tensors
        x_raw = batch['X_raw'].detach().cpu().numpy()
        x_pred = outputs['prediction'].detach().cpu().numpy()

        # Select random sample from batch
        idx_sample = np.random.randint(0, x_raw.shape[0]-1)

        break # Only need one batch for visualization

    plt.figure(figsize=(12, 6))
    plt.plot(x_raw[idx_sample], label='Raw Signal', alpha=0.7)
    plt.plot(x_pred[idx_sample], label='Reconstructed Signal', alpha=0.7)
    plt.title(f'Signal Reconstruction | Loss: {test_loss:.4f}')
    plt.xlabel('Frequency bins')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid()
    plt.savefig(run_dir / "log" / "reconstruction.png")

def save_global_metrics(run_dir:Path, best_valid_loss:float, last_valid_loss:float, best_train_loss:float, last_train_loss:float):
    """
    Save global metrics (best and last losses) to a JSON file.
    Args:
        run_dir (Path): Directory to save the metrics.
        best_valid_loss (float): Best validation loss achieved during training.
        last_valid_loss (float): Validation loss at the end of training.
        best_train_loss (float): Best training loss achieved during training.
        last_train_loss (float): Training loss at the end of training.
    """
    # Get the method and dataset from the run directory structure
    ht = run_dir.parts
    if len(ht) < 4:
        print("Unexpected run directory structure. Cannot extract method and dataset.")
        return

    
    # Create results dictionary with method, dataset, timestamp, and metrics
    results_dict = {
        "method":ht[-3],
        "dataset":ht[-2],
        "timestamp":ht[-1],
        "best_valid_loss": best_valid_loss,
        "last_valid_loss": last_valid_loss,
        "best_train_loss": best_train_loss,
        "last_train_loss": last_train_loss,
    }
    
    # Save to CSV file
    filepath = Path("results/pretrain/pretrain_metrics.csv")
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results_dict)

# Core pretraining loop
def train(
        model:nn.Module,
        train_loader:torch.utils.data.DataLoader,
        valid_loader:torch.utils.data.DataLoader,
        test_loader:torch.utils.data.DataLoader,
        device:torch.device,
        epochs:int,
        optimizer:torch.optim.Optimizer,
        args_dataloader:dict,
        args_backbone:dict,
        args_ssl:dict,
        args_training:dict,
        scheduler:None,
        evaluation:bool=False,
        ):
    """
    Pretraining loop for self-supervised learning models.
    Args:
        model (nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        valid_loader (DataLoader): DataLoader for validation data.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): Device to run the training on.
        epochs (int): Number of epochs to train.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler or None): Learning rate scheduler.
        evaluation (bool): Whether to evaluate the model after training.
        args_dataloader (dict): Arguments to pass to the dataloader.
        args_backbone (dict): Arguments to pass to the backbone.
        args_ssl (dict): Arguments to pass to the ssl method.
        args_training (dict): Arguments to pass to the training loop.
    Returns:
        None
    """
    # Create run directory
    run_dir = create_run_dir(method=model.__class__.__name__, dataset=train_loader.dataset.dataset.__class__.__name__) 
    
    # Move model to device
    model.to(device)
    best_valid_loss = float('inf')

    last_valid_loss = float('inf')
    last_train_loss = float('inf')

    best_valid_loss = float('inf')
    best_train_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training", leave=False):
            inputs = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = model.compute_loss(outputs, inputs)

            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

            # For IJEPA, update the target encoder with momentum after each batch
            if isinstance(model, IJEPAModel):
                # Update target encoder with momentum
                model.update_target_encoder()

        train_loss /= len(train_loader)

        if train_loss < best_train_loss:
            best_train_loss = train_loss

        # Validation phase
        model.eval()
        valid_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"Epoch {epoch}/{epochs} - Validation", leave=False):
                inputs = move_batch_to_device(batch, device)
                outputs = model(inputs)
                
                loss = model.compute_loss(outputs, inputs)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

        if run_dir:
            log_metrics(run_dir, epoch, train_loss, valid_loss)

        # Save best model
        if run_dir and (epoch == 1 or valid_loss < best_valid_loss):
            best_valid_loss = valid_loss
            save_model_checkpoint(run_dir, model, name="best.pt")

    last_valid_loss = valid_loss
    last_train_loss = train_loss
    
    # Save final model
    if run_dir:
        save_model_checkpoint(run_dir, model, name="last.pt")
        save_model_config(run_dir, args_dataloader, args_backbone, args_ssl, args_training) # Save config file
        plot_metrics(run_dir) # Plot training metrics

    print("Training complete.")

    # Save global metrics to CSV
    save_global_metrics(run_dir, best_valid_loss, last_valid_loss, best_train_loss, last_train_loss)

    # Evaluate on test set
    if evaluation:
        evaluate(
            run_dir=run_dir,
            device=device,
            model=model,
            test_loader=test_loader,
        )