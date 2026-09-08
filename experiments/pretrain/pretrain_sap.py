import torch, random
import numpy as np
import argparse

from models.ssl.sap import SAPModel
from dataset import dataloader
from training.pretrain import train, evaluate
from types import SimpleNamespace
from models.backbone.registry import get_backbone

def main(args):
    """
    Pretrain SAP model for self-supervised learning on time series data.
    Args:
       args (argparse): Arguments for the
          pretrain SAP model.
             - dataset (str): Dataset to pretrain.
             - window_size (int): Size of time serie window.
             - window_stride (int): Stride of time serie window.
             - batch_size (int): Batch size for datal
             - downstreaming_factor (int): Downstream factor
             - model (str): Model to pretrain.
             - learning_rate (float): Learning rate for Adam
             - weight_decay (float): Weight decay for Adam
             - epochs (int): Number of epochs to train
    """
    print("The deownsampling factor is : ", args.downsampling_factor, "with type of : ", type(args.downsampling_factor))

    # Set dataloader arguments
    args_dataloader = SimpleNamespace(
        name=args.pretrain_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,
        batch_size=args.batch_size,
        downsampling_factor=args.downsampling_factor,
        symetric_spectrum=args.symetric_spectrum,
        )
    
    # Set backbone arguments
    args_backbone = SimpleNamespace(
        model="vit1d",
        domain="frequency", # Use frequency-based learning method
        normalization="z-score", # Use z-score normalization
    )
    # Set ssl arguments
    args_ssl = SimpleNamespace(
        method="sap",
        downsampling_factor=args.downsampling_factor,
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
    model = SAPModel(backbone, args_ssl)

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
    parser = argparse.ArgumentParser(description="Pretrain SAP Model")
    
    # Dataloader
    parser.add_argument('--pretrain_dataset', type=str, required=True, choices=['CWRU','LASPI'], help='Name of the dataset to use for pretraining')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0003695, help='Learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1.1133e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')

    # # SAP specific
    parser.add_argument('--downsampling_factor', required=True, help='Downsampling factor')
    parser.add_argument('--symetric_spectrum', action='store_true', help='Whether to use symetric spectrum for SAP')
    
    args = parser.parse_args()

    # # Debugging arguments
    # args = SimpleNamespace(
    #     pretrain_dataset="CWRU",
    #     downstream_dataset="CWRU",
    #     batch_size=256,
    #     window_size=2048,
    #     window_stride=256,
    #     learning_rate=0.0003695,
    #     weight_decay=1.1133e-5,
    #     epochs=5,
    #     downsampling_factor=None,
    # )

    main(args)