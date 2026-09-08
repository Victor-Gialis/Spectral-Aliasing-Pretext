import torch, os
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
from models.ssl.ijepa import IJEPAModel

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
    
    # Split dataloaders
    train_loader, valid_loader, test_loader, labels, dataset = split_data_factory.split_dataloader(
        split_type=args.split_type,
        args_dataloader=args_dataloader,
    )

    # Backbone
    backbone_random = ViT1DEncoder()
    ssl_mode = IJEPAModel(backbone=backbone_random
                        , args_ssl=SimpleNamespace(enc_mask_ratio=1.0, pred_mask_ratio=1.0, momentum=1.0)
    )
    
    if args.backbone_checkpoint_path is None:
        # Get last checkpoint for IJEPAModel pretrained on CWRU
        checkpoints = os.listdir("results/pretrain/IJEPAModel/CWRUDataset")
        checkpoints.sort()

        print("Available checkpoints:", checkpoints)
        checkpoint_path = os.path.join("results/pretrain/IJEPAModel/CWRUDataset", checkpoints[-1])  # Get the last checkpoint
    else:
        # Get checkpoint from specified path
        checkpoint_path = args.backbone_checkpoint_path

    # checkpoint_path = "results/pretrain/IJEPAModel/CWRUDataset/20260123_162146"
    ssl_model = load_model_checkpoint(ssl_mode, checkpoint_path)

    # Load the pretrained backbone from the SSL model
    args.enc_mask_ratio = ssl_model.enc_mask_ratio
    args.pred_mask_ratio = ssl_model.pred_mask_ratio
    args.momentum = ssl_model.momentum
    
    backbone = ssl_model.backbone

    # Downstream model
    model = get_downstream_model(
        backbone=backbone,
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
    parser = argparse.ArgumentParser(description="Downstream IJEPA Model")

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
    parser.add_argument('--backbone_checkpoint_path', type=str, default=None, help='Path to the pretrained backbone model checkpoint (if not using SAP)')

    args = parser.parse_args()
    args.backbone_init = "ijepa"

    # Run experiment
    main(args)
