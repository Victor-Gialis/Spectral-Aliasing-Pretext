import torch, random
import numpy as np
import argparse

from models.ssl.sajepa import SAJEPAModel
from dataset import dataloader
from training.pretrain import train, evaluate
from types import SimpleNamespace
from models.backbone.registry import get_backbone

def main(args):
    """
    Pretrain SA-JEPA model for self-supervised learning on time series data.
    Args:
       args (argparse): Arguments for the
          pretrain SA-JEPA model.
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

    # Set dataloader arguments
    args_dataloader = SimpleNamespace(
        name=args.pretrain_dataset,
        window_size=args.window_size,
        window_stride=args.window_stride,   
        batch_size=args.batch_size,
        downsampling_factor=args.downsampling_factor,
        symetric_spectrum=args.symetric_spectrum,
        scaling_method=args.scaling_method,
        )
    
    # Set backbone arguments
    args_backbone = SimpleNamespace(
        model="vit1d",
        domain=args.domain, # Use frequency-based learning method
        scaling_method=args.scaling_method, # Use scaling method
    )
    # Set ssl arguments
    args_ssl = SimpleNamespace(
        method="sajepa",
        momentum=args.momentum,
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
    if len(args.pretrain_dataset) > 1 :
        train_loader, valid_loader, test_loader, dataset = dataloader.get_multi_dataset_ssl_splits(args_dataloader)
    else :
        args_dataloader.name = args.pretrain_dataset[0]
        train_loader, valid_loader, test_loader = dataloader.get_heterogeneous_split_dataloaders(args_dataloader)

    # Initialize backbone
    backbone = get_backbone(args_backbone).to(device)
    args_ssl.args = vars(backbone._get_arguments())

    # Initialize SAJEPA model
    model = SAJEPAModel(backbone, args_ssl).to(device)

    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args_training.learning_rate, weight_decay=args_training.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Set momentum scheduler for target encoder
    ipe = len(train_loader)
    ipe_scale = 1.0 
    n_epochs = args_training.epochs

    momentum_scheduler = [(args_ssl.momentum + i*(1.0 - args_ssl.momentum)) / (ipe * n_epochs * ipe_scale) for i in range(int(ipe * n_epochs * ipe_scale))]
    model._load_momentum_scheduler(momentum_scheduler)

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
    parser = argparse.ArgumentParser(description="Pretrain SA-JEPA model for self-supervised learning on time series data.")
    
    # # Dataloader
    parser.add_argument('--pretrain_dataset', 
                type=str, 
                nargs='+', 
                required=True, 
                help='Name of the dataset(s) to use for pretraining')
    
    parser.add_argument("--window_size", type=int, default=2048, help="Size of time series window.")
    parser.add_argument("--window_stride", type=int, default=512, help="Stride of time series window.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for dataloader.")    

    parser.add_argument('--domain', type=str, default='frequency', choices=['frequency','time'], help='Domain for the backbone model (time or frequency).')
    parser.add_argument('--scaling_method', type=str, default='standardization', choices=[None,'normalization','standardization'], help='Scaling method for the dataset')

    # Training
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
        
    # SAJEPA specific
    parser.add_argument('--downsampling_factor', required=True, help='Downsampling factor')
    parser.add_argument('--symetric_spectrum', action='store_true', help='Whether to use symetric spectrum for SAP')
    parser.add_argument("--momentum", type=float, default=0.97, help="Momentum for target encoder in SA-JEPA.")

    args = parser.parse_args()
    main(args)