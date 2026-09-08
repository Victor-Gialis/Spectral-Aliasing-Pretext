import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace

from dataset import dataloader, split_data_factory
from training.downstream import train, evaluate, log_metrics
from training.pretrain import load_model_checkpoint
from models.downstream.registry import get_downstream_model
from models.backbone.vit1d import ViT1DEncoder
from models.backbone.registry import get_backbone
from models.ssl.registry import get_pretrained_backbone
from models.ssl.sap import SAPModel

def main(args):
    """
    Run ONE data-scarcity downstream experiment.
    Args:
        args (object): Arguments for the experiment.
    """

    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloader configuration
    args_dataloader = SimpleNamespace(
        name=args.downstream_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_ratio=args.data_ratio,
        seed=args.seed,
    )
    
    # Get same train loader as pretrain
    pretrain_args_dataloader = SimpleNamespace(
        name=args.pretrain_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        data_ratio=1.0,  # Use full data for pretraining
        seed=args.seed,
    )

    # Load pretrain train loader
    pretrain_loader, _, _ = dataloader.get_heterogeneous_split_dataloaders(pretrain_args_dataloader) # Get train loader only

    # Backbone - random init
    backbone_random = ViT1DEncoder()

    # Compute stats from X_raw train dataloader
    stats = dataloader.compute_stats_from_dataloader(pretrain_loader)
    backbone_random._loads_stats(stats)

    # Split dataloaders
    train_loader, valid_loader, test_loader, labels, dataset = split_data_factory.split_dataloader(
        split_type=args.split_type,
        args_dataloader=args_dataloader,
    )

    # Downstream model
    model = get_downstream_model(
        backbone=backbone_random,
        task=args.task, 
        head_type=args.head_type,# "classification" | "regression"
        classes=labels,
        freeze_backbone= not args.finetune,
        device=device
    )

    # Optimisation
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    # Training
    train(
        args=args,
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=args.epochs,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downstream Random Model")

    # Dataloader
    parser.add_argument('--pretrain_dataset', type=str, required=True, choices=['CWRU','LASPI','CVRTEST'], help='Name of the dataset to use for pretraining')
    parser.add_argument('--downstream_dataset', type=str, required=True, choices=['CWRU','LASPI','CVRTEST'], help='Name of the dataset to use for downstream task')

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    parser.add_argument('--data_ratio', type=float, required=True, help='Ratio of training data to use for the downstream task')
    parser.add_argument('--split_type', type=str, required=True, choices=["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"], help='Type of data split to use for the downstream task (independent, speed_stratified, speed_load_stratified, sample_stratified)')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0003695, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1.1133e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')

    # Backbone
    parser.add_argument('--head_type', type=str, required=True, choices=["linear", "nonlinear"], help='Type of head to use for downstream model (linear or nonlinear)')
    parser.add_argument('--finetune', action='store_true', help='Whether to finetune the backbone during downstream training')
    parser.add_argument('--task', type=str, required=True, choices=["classification", "regression"], help='Type of downstream task (classification or regression)')
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')

    args = parser.parse_args()

    # # Debug run
    # args = parser.parse_args(args=[
    #     "--pretrain_dataset", "CWRU",
    #     "--downstream_dataset", "CWRU",
    #     "--batch_size", "64",             
    #     "--window_size", "2048",
    #     "--window_stride", "256",
    #     "--data_ratio", "0.01",
    #     "--split_type", "speed_stratified",
    #     "--learning_rate", "0.0003695",
    #     "--weight_decay", "1.1133e-5",
    #     "--epochs", "2",
    #     "--head_type", "linear",
    #     "--task", "classification",
    #     "--seed", "0",
    #     "--finetune"
    # ])

    args.backbone_init = "random"

    # Run experiment
    main(args)