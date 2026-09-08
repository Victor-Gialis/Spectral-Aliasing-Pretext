import torch, random
import numpy as np
import argparse

from models.ssl.mae import MAEModel
from dataset import dataloader
from training.pretrain import train, evaluate
from types import SimpleNamespace
from models.backbone.registry import get_backbone

def main(args):
    # Set dataloader arguments
    args_dataloader = SimpleNamespace(
        name=args.pretrain_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        )
    # Set backbone arguments
    args_backbone = SimpleNamespace(
        model="vit1d",
        domain="frequency", # Use frequency-based learning method
        normalization="z-score", # Use z-score normalization
    )
    # Set ssl arguments
    args_ssl = SimpleNamespace(
        method="mae",
        mask_ratio=args.mask_ratio,
    )
    
    # Set training arguments
    args_training = SimpleNamespace(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
    )

    # Set seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare data loaders
    train_loader, valid_loader, test_loader = dataloader.get_heterogeneous_split_dataloaders(args_dataloader)

    # Initialize backbone
    backbone = get_backbone(args_backbone).to(device)
    args_ssl.args = vars(backbone._get_arguments())

    # Compute min-max from X_raw train dataloader
    stats = dataloader.compute_stats_from_dataloader(train_loader)
    backbone._loads_stats(stats)

    # Initialize ssl method
    model = MAEModel(backbone, args_ssl)

    # Define optimizer and scheduler    
    optimizer = torch.optim.AdamW(model.parameters(), lr= args_training .learning_rate, weight_decay=args_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_training .epochs, eta_min=1e-6)
        
    # Start training
    train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        device=device,
        epochs=args_training.epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        args_dataloader=args_dataloader,
        args_backbone=args_backbone,
        args_training=args_training,
        args_ssl=args_ssl,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pretrain MAE Model")
    
    # Dataloader
    parser.add_argument('--pretrain_dataset', type=str, required=True, choices=['CWRU','LASPI'], help='Name of the dataset to use for pretraining')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0003695, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1.1133e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')

    # MAE specific
    parser.add_argument('--mask_ratio', type=float, required=True, help='Masking ratio for MAE')
    
    args = parser.parse_args()
    main(args)
