import csv,os
import torch, json
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from models.backbone.vit1d import ViT1DEncoder
from models.ssl.mae import MAEModel

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from matplotlib.patches import Rectangle
from dataset.transform import normalization

def create_run_dir(base_dir="results/downstream", args:object=None):
    """
    Docstring for create_run_dir
    
    :param base_dir: Description
    :param args: Description
    :type args: object
    """ 
    split_type = args.split_type
    pretrain_dataset = args.pretrain_dataset 
    downstream_dataset = args.downstream_dataset
    backbone_init = args.backbone_init
    head_type = args.head_type
    finetune = args.finetune
    seed = args.seed

    # create split directory
    split_dir = os.path.join(base_dir, split_type)
    os.makedirs(split_dir, exist_ok=True)

    if backbone_init == "random":
        pretrain_dataset = "None"

    # create directory structure
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir = f"{pretrain_dataset}_to_{downstream_dataset}_backbone_{backbone_init}_head_{head_type}_finetune_{finetune}"
    run = os.path.join(split_dir, dir, timestamp)
    os.makedirs(run, exist_ok=True)

    data_ratio = args.data_ratio
    epochs = args.epochs

    # create sub-directory for data ratio and epochs
    dir_name = f"data_ratio_{data_ratio}_epochs_{epochs}_seed_{seed}"
    run_dir = os.path.join(run, dir_name)
    os.makedirs(run_dir, exist_ok=True)

    os.makedirs(os.path.join(run_dir, "checkpoint"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "log"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "attention_visualizations"), exist_ok=True)

    return run_dir

def save_model_config(run_dir, args):
    ### save the model configuration in json format
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)
    
def save_model_checkpoint(model, run_dir, name="model_checkpoint.pth"):
    """
    Save model checkpoint with backbone stats if available.
    
    :param model: The model to save
    :param run_dir: Directory to save the checkpoint
    :param name: Name of the checkpoint file
    """
    checkpoint_path = os.path.join(run_dir, "checkpoint", name)
    
    # Create checkpoint dictionary with model weights
    checkpoint = {
        'model_state_dict': model.state_dict(),
    }
    
    # Save backbone stats if available
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'stats'):
        checkpoint['backbone_stats'] = model.backbone.stats
    
    torch.save(checkpoint, checkpoint_path)

def load_model_checkpoint(model, checkpoint_path, device="cpu", strict=True):
    """
    Load model checkpoint and restore backbone stats if available.
    
    :param model: The model to load weights into
    :param checkpoint_path: Path to the checkpoint file
    :param device: Device to load the model on
    :param strict: If False, ignores missing and unexpected keys
    :return: The model with loaded weights and stats
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both old format (just state_dict) and new format (checkpoint dict)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        # Restore backbone stats if available
        if 'backbone_stats' in checkpoint and hasattr(model, 'backbone'):
            model.backbone.stats = checkpoint['backbone_stats']
    else:
        # Legacy format: checkpoint is just state_dict
        # Try with strict=True first, then with strict=False if it fails
        try:
            model.load_state_dict(checkpoint, strict=strict)
        except RuntimeError as e:
            if strict:
                print(f"⚠️  Loading checkpoint with strict=False due to: {str(e)[:100]}...")
                model.load_state_dict(checkpoint, strict=False)
            else:
                raise
    
    return model

def plot_metrics(train_losses, valid_losses, run_dir):
    """
    Docstring for plot_metrics
    
    :param train_losses: Description
    :param valid_losses: Description
    :param run_dir: Description
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(run_dir, "log","loss_curve.png"))
    plt.close()

def plot_test_metrics_over_epochs(all_test_metrics, run_dir, task="classification"):
    """
    Plot test metrics over epochs.
    
    :param all_test_metrics: List of metric dictionaries from each epoch
    :param run_dir: Directory to save the plots
    :param task: Task type (classification or regression)
    """
    if not all_test_metrics:
        return
    
    epochs = range(1, len(all_test_metrics) + 1)
    
    if task == "classification":
        accuracies = [m.get("accuracy", 0) for m in all_test_metrics]
        f1_scores = [m.get("f1", 0) for m in all_test_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, accuracies, marker='o', label="Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Test Accuracy over Epochs")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, f1_scores, marker='s', label="F1-Score", color='orange')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1-Score")
        ax2.set_title("Test F1-Score over Epochs")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "test_metrics_over_epochs.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    else:  # regression
        mses = [m.get("mse", 0) for m in all_test_metrics]
        r2s = [m.get("r2", 0) for m in all_test_metrics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.plot(epochs, mses, marker='o', label="MSE")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE")
        ax1.set_title("Test MSE over Epochs")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(epochs, r2s, marker='s', label="R²", color='orange')
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("R² Score")
        ax2.set_title("Test R² Score over Epochs")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "test_metrics_over_epochs.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_attention_spectra(run_dir, model, test_loader, device, task="classification", nyquist_freq=6000):
    """
    Plot average spectrum for each class with patches colored by attention scores.
    
    :param run_dir: Directory to save the plots
    :param model: Model with backbone that has get_attention_scores()
    :param test_loader: Test data loader
    :param device: Device to run on
    :param task: Task type (classification or regression)
    """
    if task != "classification":
        return
    
    if not isinstance(model.backbone, ViT1DEncoder):
        return
    
    model.eval()
    class_spectra = {}
    class_attention_scores = {}
    df_top_attention = list()

    # Create run_dir if not exist
    os.makedirs(run_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting Attention Scores", leave=False):
            inputs = move_batch_to_device(batch, device)

            # Get input data            
            if model.backbone.domain == "time":
                X_raw = inputs['x_raw']
            
            elif model.backbone.domain == "frequency":
                X_raw = inputs['X_raw']

            else:
                raise ValueError(f"Unsupported backbone domain: {model.backbone.domain}")
        
            # If stats normalise
            if hasattr(model.backbone, 'stats') and model.backbone.stats is not None:
                # Normalize (same as in downstream model)
                x_norm = normalization.global_z_log_normalization(x=X_raw, stats=model.backbone.stats)
            else:
                x_norm = X_raw
            
            # Get attention scores from backbone for each sample
            batch_attention_scores = []
            for i in range(x_norm.shape[0]):
                single_sample = x_norm[i:i+1]  # (1, n_steps)
                attn_score = model.backbone.get_attention_scores(single_sample)  # (n_patches,)
                batch_attention_scores.append(attn_score.cpu().numpy())
            
            batch_attention_scores = np.array(batch_attention_scores)  # (batch, n_patches)
            
            # Get forward pass for class labels
            outputs = model(inputs)
            targets = model.head.transform_labels(inputs['y_label'])
            class_labels = torch.argmax(targets, dim=1).cpu().numpy()
            
            # Get input spectra
            X = X_raw.cpu().numpy()  # (batch, n_steps)
            
            # Aggregate by class
            for i, class_idx in enumerate(class_labels):
                class_idx = int(class_idx)
                if class_idx not in class_spectra:
                    class_spectra[class_idx] = []
                    class_attention_scores[class_idx] = []
                
                class_spectra[class_idx].append(X[i])
                class_attention_scores[class_idx].append(batch_attention_scores[i])
    
    # Compute mean spectra and attention scores per class
    for class_idx in tqdm(sorted(class_spectra.keys()), desc="Plotting Attention Spectra", leave=False):
        spectra = np.array(class_spectra[class_idx])  # (n_samples, n_steps)
        attention_scores = np.array(class_attention_scores[class_idx])  # (n_samples, n_patches)
        
        mean_spectrum = spectra.mean(axis=0)  # (n_steps,)
        mean_attention = attention_scores.mean(axis=0)  # (n_patches,) - mean attention per patch
        mean_attention = mean_attention.squeeze()  # (n_patches,)
        
        # Plot single spectrum with patches colored by attention
        n_patches = mean_attention.size
        
        fig, ax = plt.subplots(figsize=(14, 4))
        
        class_name = model.head.lb.classes_[class_idx]
        
        # Normalize attention scores to [0, 1] for coloring
        norm_attention = (mean_attention - mean_attention.min()) / (mean_attention.max() - mean_attention.min() + 1e-6)
        norm_attention = norm_attention.squeeze()  # (n_patches,)

        # Plot spectrum with patches colored by attention scores
        n_steps = len(mean_spectrum)
        patch_width = n_steps / n_patches
        frequency_bins = np.arange(n_steps)*(nyquist_freq/n_steps)
        
        cmap = plt.cm.Reds

        # Get top-5 attention patches
        top_patches_idx = np.argsort(mean_attention)[-5:]
        start_idx_freq = top_patches_idx * int(patch_width)-1
        end_idx_freq = (top_patches_idx + 1) * int(patch_width)-1
        top_attention_score = mean_attention[top_patches_idx]

        df_top_attention.append(pd.DataFrame({
            'Classes': [class_name for _ in range(len(top_patches_idx))],
            'Patch Index': top_patches_idx,
            'Frequency Range': [f"{frequency_bins[start]:.1f}-{frequency_bins[end]:.1f} Hz" for start,end in zip(start_idx_freq, end_idx_freq)],
            'Attention Score': top_attention_score
        }).sort_values(by='Attention Score', ascending=False))

        # print(f"Attention score shape: {mean_attention.shape}, Mean spectrum shape: {mean_spectrum.shape}")
        for patch_idx, att_score in np.ndenumerate(mean_attention):
            patch_idx = patch_idx[0]
            start = int(patch_idx * patch_width)
            end = int((patch_idx + 1) * patch_width)
            
            if start < n_steps and end <= n_steps:
                ax.fill_between(frequency_bins[start:end], mean_spectrum[start:end], alpha=0.7, 
                               color=cmap(att_score))
        
        ax.plot(frequency_bins, mean_spectrum, 'k-', linewidth=0.5, alpha=0.3)
        ax.set_xlabel('Frequency (Hz)', fontweight='bold')
        ax.set_ylabel('Amplitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.suptitle(f'Mean Spectrum - Class: {class_name}\n(Patches colored by mean attention scores)', 
                     fontsize=14, fontweight='bold', y=0.995)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=plt.Normalize(vmin=mean_attention.min(), vmax=mean_attention.max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Attention Score', fontweight='bold')
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(run_dir,f'attention_spectrum_class_{class_name}.png'), 
                   dpi=300, bbox_inches='tight')

        plt.close()

    return df_top_attention

def evaluate(run_dir, model, test_loader, device, task="classification"):
    """
    Docstring for evaluate
    
    :param model: Description
    :param test_loader: Description
    :param device: Description
    :param task: Description
    """
    model.eval()
    all_preds = []
    all_labels = []

    # Iterate over test data and collect predictions and labels
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating", leave=False):
            inputs = move_batch_to_device(batch, device)
            outputs = model(inputs)

            # For classification, we assume the model's head outputs logits for each class.
            # For regression, we assume the model's head outputs a single value.
            if task == "classification":
                targets = model.head.transform_labels(inputs['y_label'])
                targets = torch.argmax(targets, dim=1)
                preds = torch.argmax(outputs, dim=1)
            else:  # regression
                preds = outputs.squeeze()
                targets = inputs['y_label'].squeeze()
                
            all_preds.append(preds.detach().cpu())
            all_labels.append(targets.detach().cpu())
    
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    metrics = {}
    if task == "classification":
        metrics["accuracy"] = accuracy_score(all_labels, all_preds)
        metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")

        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds, normalize='true')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.head.lb.classes_)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, cmap='Reds', values_format='.1%')
        
        # Améliorer la lisibilité
        ax.set_xlabel('Prédiction', fontsize=12, fontweight='bold')
        ax.set_ylabel('Vérité', fontsize=12, fontweight='bold')
        ax.set_title('Matrice de Confusion', fontsize=14, fontweight='bold', pad=20)
        
        # Augmenter la taille des labels
        ax.xaxis.set_tick_params(labelsize=11)
        ax.yaxis.set_tick_params(labelsize=11)
        
        # Rotation des labels pour éviter le chevauchement
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Sauvegarder la figure
        os.makedirs(os.path.join(run_dir, "log"), exist_ok=True)
        
        plt.savefig(os.path.join(run_dir,'log', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    else:
        metrics["mse"] = mean_squared_error(all_labels, all_preds)
        metrics["r2"] = r2_score(all_labels, all_preds)

    return metrics

def log_metrics(run_dir, results_dict):
    global_dir = "results/downstream"
    filename = "metrics.csv"

    run_filepath = os.path.join(run_dir, "log", filename)
    global_filepath = os.path.join(global_dir, filename)

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(global_dir, exist_ok=True)
    
    for filepath in [run_filepath, global_filepath]:
        file_exists = os.path.isfile(filepath)
        
        with open(filepath, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(results_dict)

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

def train(
    args:object,
    model:nn.Module,
    train_loader:torch.utils.data.DataLoader,
    valid_loader:torch.utils.data.DataLoader,
    test_loader:torch.utils.data.DataLoader,
    device:torch.device,
    optimizer,
    scheduler,
    epochs,
    eval_every_epoch=False,
):    
    # Create run directory
    run_dir = create_run_dir(base_dir="results/downstream", args=args)

    all_train_losses = []
    all_valid_losses = []
    all_test_metrics = []  # Store test metrics for each epoch if eval_every_epoch=True
    
    # Move model to device
    model.to(device)
    best_valid_loss = float("inf")

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

        train_loss /= len(train_loader)
        all_train_losses.append(train_loss)

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
        all_valid_losses.append(valid_loss)

        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model_checkpoint(model, run_dir, name="best_model.pth")

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")
        
        # Evaluate on test set if eval_every_epoch is True
        if eval_every_epoch:
            metrics = evaluate(run_dir, model, test_loader, device, task=args.task)
            all_test_metrics.append(metrics)
            print(f"  Test Metrics: {metrics}")

    # Plot training curves
    if run_dir:
        plot_metrics(all_train_losses, all_valid_losses, run_dir)
        
        # Plot test metrics over epochs if eval_every_epoch is True
        if eval_every_epoch:
            plot_test_metrics_over_epochs(all_test_metrics, run_dir, task=args.task)
    
    # Save last model
    save_model_checkpoint(model, run_dir, name="last_model.pth")
    
    # Plot attention spectra for each class
    plot_attention_spectra(run_dir, model, test_loader, device, task=args.task)
    
    # Testing phase (only if not already evaluated every epoch)
    if not eval_every_epoch:
        metrics = evaluate(run_dir, model, test_loader, device, task=args.task)
    else:
        # Use the last epoch metrics
        metrics = all_test_metrics[-1]

    # Préparer le dictionnaire pour le CSV
    results_dict = {
        "pretrain_dataset":args.pretrain_dataset,
        "downstream_dataset": args.downstream_dataset,
        "epochs":args.epochs,
        "backbone_init": args.backbone_init,
        "head_type": args.head_type,
        "finetune":args.finetune,
        "seed": args.seed,
        "data_ratio": args.data_ratio,
        "mask_ratio": "None",
        "downsample_factor": "None",
    }

    # Add mask ratio if the backbone is MAE
    if args.backbone_init == "mae":
        results_dict["mask_ratio"] = args.mask_ratio if hasattr(args, "mask_ratio") else "None"
    
    # Add downsampling factor if it exists
    if args.backbone_init == "sap":
        results_dict["downsample_factor"] = args.downsample_factor if hasattr(args, "downsample_factor") else "None"

    results_dict.update(metrics)

    # Sauvegarde de la configuration du modèle
    save_model_config(run_dir, args)

    # Sauvegarde dans CSV
    log_metrics(run_dir, results_dict)
    print(f"Results added to csv : {results_dict}")