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
             - scaling_method (str): Scaling method (normalization or standardization).
             - model (str): Model to pretrain.
             - learning_rate (float): Learning rate for Adam
             - weight_decay (float): Weight decay for Adam
             - epochs (int): Number of epochs to train
    """
    print("The deownsampling factor is : ", args.downsampling_factor, "with type of : ", type(args.downsampling_factor))

    if args.frequency_positions:
        if args.domain != 'frequency':
            raise ValueError('--frequency_positions requires --domain frequency')
        try:
            factor = int(args.downsampling_factor)
        except (TypeError, ValueError) as exc:
            raise ValueError('--frequency_positions requires a fixed integer downsampling factor') from exc
        if not 1 <= factor <= args.window_size // 2 or args.window_size % 32:
            raise ValueError('Use a window size divisible by 32 and a factor between 1 and half the window size')
        if not args.symetric_spectrum and (args.window_size + factor - 1) // factor < 32:
            raise ValueError('The folded spectrum must contain at least one patch')

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
        frequency_positions=args.frequency_positions,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        heads=args.heads,
        domain=args.domain, # Use frequency-based learning method
        scaling_method=args.scaling_method, # Use None for normalization stats
    )
    # Set ssl arguments
    args_ssl = SimpleNamespace(
        method="sap",
        downsampling_factor=args.downsampling_factor,
        symetric_spectrum=args.symetric_spectrum,
        loss_domain=args.loss_domain,
        revin=args.revin,
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
    if len(args.pretrain_dataset) > 1 : # Composite dataset
        train_loader, valid_loader, test_loader, _ = dataloader.get_multi_dataset_ssl_splits(args_dataloader)
    else :
        train_loader, valid_loader, test_loader = dataloader.get_heterogeneous_split_dataloaders(args_dataloader)

    # Initialize backbone
    backbone = get_backbone(args_backbone).to(device)
    args_ssl.args = vars(backbone._get_arguments()) # Store backbone arguments in args_ssl for later use

    # Initialize ssl method
    model = SAPModel(backbone, args_ssl).to(device)

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
    parser.add_argument('--pretrain_dataset', 
                    type=str, 
                    nargs='+', 
                    required=True, 
                    help='Name of the dataset(s) to use for pretraining')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--window_size', type=int, default=2048, help='Window size for data segments')
    parser.add_argument('--window_stride', type=int, default=256, help='Stride for windowing data segments')

    parser.add_argument('--domain', type=str, default='frequency', choices=['frequency','time'], help='Domain for the backbone model (time or frequency).')
    parser.add_argument('--scaling_method', type=str, default='standardization', choices=[None,'normalization','standardization'], help='Scaling method for the dataset')

    # Training
    parser.add_argument('--learning_rate', type=float, default=0.0003695, help='Learning rattimee for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1.1133e-5, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')

    # Backbone specific
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the backbone model')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of layers for the backbone model')
    parser.add_argument('--heads', type=int, default=4, help='Number of attention heads for the backbone model')

    # SAP specific
    parser.add_argument('--downsampling_factor', required=True, help='Downsampling factor')
    parser.add_argument('--frequency_positions', action='store_true',
                        help='Use physical frequency and bandwidth as positional coordinates')
    parser.add_argument('--symetric_spectrum', action='store_true', help='Whether to use symetric spectrum for SAP')
    parser.add_argument('--loss_domain', type=str, default='linear', choices=['linear','log'], help='Loss domain for SAP (linear or log)')
    parser.add_argument('--revin', action='store_true', help='Whether to use Reversible Instance Normalization (ReVIN) for SAP')
    
    args = parser.parse_args()
    main(args)