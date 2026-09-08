import torch, random
import numpy as np
import argparse

from models.ssl.ts2vec import TS2VecModel
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
        model="dilated_cnn", 
        domain="time", # Use time-based learning method
        normalization="z-score", # Use z-score normalization

    )
    # Set ssl arguments
    args_ssl = SimpleNamespace(
        method="ts2vec",
        temperature=args.temperature,
        use_cosine_similarity=args.use_cosine_similarity,
        projection_dim=args.projection_dim,
        temporal_unit=args.temporal_unit,
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

    # # Compute min-max from X_raw train dataloader
    stats = dataloader.compute_stats_from_dataloader(train_loader)
    backbone._loads_stats(stats)

    # Initialize ssl method
    model = TS2VecModel(backbone, args_ssl).to(device)

    # Define optimizer and scheduler    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args_training.learning_rate, weight_decay=args_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args_training.epochs, eta_min=1e-6)
        
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
    parser = argparse.ArgumentParser(description="Pretrain TS2Vec Model")
    
    # Dataloader
    parser.add_argument('--pretrain_dataset', type=str, default='CWRU', choices=['CWRU','LASPI'], help='Name of the dataset to use for pretraining')
    
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    # Training
    parser.add_argument('--learning_rate', type=float, default=3e-4, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')

    # TS2Vec specific
    parser.add_argument('--temperature', type=float, default=0.2, help='Temperature parameter for contrastive loss')
    parser.add_argument('--projection_dim', type=int, default=64, help='Dimension of projection head output')
    parser.add_argument('--temporal_unit', type=int, default=0, help='Temporal unit for instance discrimination')
    parser.add_argument('--use_cosine_similarity', action='store_true', help='Use cosine similarity instead of dot product for contrastive loss')
    
    args = parser.parse_args()
    main(args)
