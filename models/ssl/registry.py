# factory SSL
import json
import torch
from pathlib import Path
from training.pretrain import load_model_checkpoint
from types import SimpleNamespace


def get_pretrained_backbone(
    method: str,
    pretrain_dataset: str,
    device: torch.device | str = "cpu",
):
    """
    Load a pretrained SSL backbone (encoder only).
    """

    from models.backbone.registry import get_backbone

    # SSL methods
    from models.ssl.mae import MAEModel
    from models.ssl.sap import SAPModel
    from models.ssl.ijepa import IJEPAModel
    from models.ssl.ts2vec import TS2VecModel

    # Datasets
    from dataset.cwru import CWRUDataset
    from dataset.laspi import LASPIDataset
    from dataset.cvrtest import CVRTESTDataset

    # -------------------------------------------------
    # Registries
    # -------------------------------------------------
    SSL_REGISTRY = {
        "mae": MAEModel,
        "sap": SAPModel,
        "ijepa": IJEPAModel,
        "ts2vec": TS2VecModel,
    }

    DATASET_REGISTRY = {
        "CWRU": CWRUDataset,
        "LASPI": LASPIDataset,
        "CVRTest": CVRTESTDataset,
    }

    assert method in SSL_REGISTRY, f"Unknown SSL method: {method}"
    assert pretrain_dataset in DATASET_REGISTRY, f"Unknown dataset: {pretrain_dataset}"

    ssl_class = SSL_REGISTRY[method]
    dataset_class = DATASET_REGISTRY[pretrain_dataset]

    # -------------------------------------------------
    # Locate run directory
    # -------------------------------------------------
    base_dir = (
        Path("results/pretrain")
        / ssl_class.__name__
        / dataset_class.__name__
    )

    if not base_dir.exists():
        raise FileNotFoundError(
            f"No pretrained model found for {method} on {dataset_class.__name__}"
        )

    # Take latest run
    run_dir = sorted(base_dir.iterdir())[-1]

    # -------------------------------------------------
    # Load config
    # -------------------------------------------------
    with open(run_dir / "config.json", "r") as f:
        config = json.load(f)

    args_backbone = SimpleNamespace(**config["backbone"])
    args_ssl = SimpleNamespace(**config["ssl"])

    # -------------------------------------------------
    # Rebuild backbone
    # -------------------------------------------------
    backbone = get_backbone(args_backbone)

    # -------------------------------------------------
    # Rebuild SSL model (only to load weights)
    # -------------------------------------------------
    ssl_model = ssl_class(backbone, args_ssl)

    #   ----------------------------------------------
    # Load weights
    #   ----------------------------------------------
    ssl_model = load_model_checkpoint(ssl_model, run_dir / "checkpoints" / "best.pt")

    return ssl_model.backbone
