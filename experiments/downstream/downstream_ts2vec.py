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
from models.backbone.cnn import DilatedCNNEncoder
from models.backbone.registry import get_backbone
from models.ssl.registry import get_pretrained_backbone
from models.ssl.ts2vec import TS2VecModel

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
    backbone_random = DilatedCNNEncoder()
    ssl_mode = TS2VecModel(backbone=backbone_random
                        , args_ssl=SimpleNamespace(temperature=1.0,
                                                   projection_dim=256,
                                                   use_cosine_similarity=True,
                                                   temporal_unit=0)
    )
    
    if args.backbone_checkpoint_path is None:
        # Get last checkpoint for TS2VecModel pretrained on CWRU
        checkpoints = os.listdir("results/pretrain/TS2VecModel/CWRUDataset")
        checkpoints.sort()

        print("Available checkpoints:", checkpoints)
        checkpoint_path = os.path.join("results/pretrain/TS2VecModel/CWRUDataset", checkpoints[-1])  # Get the last checkpoint
    else:
        # Get checkpoint from specified path
        checkpoint_path = args.backbone_checkpoint_path

    # checkpoint_path = "results/pretrain/TS2VecModel/CWRUDataset/20260123_162146"
    ssl_model = load_model_checkpoint(ssl_mode, checkpoint_path)

    # Load the pretrained backbone from the SSL model
    args.temperature = ssl_model.temperature
    args.projection_dim = ssl_model.projection_dim
    args.temporal_unit = ssl_model.temporal_unit
    args.use_cosine_similarity = ssl_model.use_cosine_similarity

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
    parser.add_argument('--pretrain_dataset', type=str, default="CWRU", choices=['CWRU','LASPI','CVRTEST'], help='Name of the dataset to use for pretraining')
    parser.add_argument('--downstream_dataset', type=str, default="CWRU", choices=['CWRU','LASPI','CVRTEST'], help='Name of the dataset to use for downstream task')

    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    parser.add_argument('--data_ratio', type=float, default=0.1, help='Ratio of training data to use for the downstream task')
    parser.add_argument('--split_type', type=str, default="speed_stratified", choices=["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"], help='Type of data split to use for the downstream task (independent, speed_stratified, speed_load_stratified, sample_stratified)')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0003695, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1.1133e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

    # Backbone
    parser.add_argument('--head_type', type=str, default="linear", choices=["linear", "nonlinear"], help='Type of head to use for downstream model (linear or nonlinear)')
    parser.add_argument('--finetune', action='store_true', help='Whether to finetune the backbone during downstream training')
    parser.add_argument('--task', type=str, default="classification", choices=["classification", "regression"], help='Type of downstream task (classification or regression)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--backbone_checkpoint_path', type=str, default=None, help='Path to the pretrained backbone model checkpoint (if not using SAP)')

    args = parser.parse_args()
    args.backbone_init = "ts2vec"

    # Run experiment
    main(args)
