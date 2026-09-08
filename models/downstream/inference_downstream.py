"""
Script d'inférence pour utiliser le modèle downstream entraîné.

Cet script montre comment:
1. Charger le modèle downstream entraîné
2. Charger les données de test
3. Faire des prédictions
4. Afficher les résultats
"""
import tqdm
import torch
import json
import os
import argparse
import numpy as np
from types import SimpleNamespace
from pathlib import Path

from dataset import split_data_factory, dataloader
from models.backbone.vit1d import ViT1DEncoder
from models.ssl.mae import MAEModel
from models.downstream.registry import get_downstream_model
from training.pretrain import load_model_checkpoint as load_pretrain_checkpoint
from training.downstream import load_model_checkpoint
from dataset.transform import normalization

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_decision_surface(model, labels, features, predictions, targets):
    """
    """
    # Réduction de dimension avec PCA
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    variance_explained = pca.explained_variance_ratio_.sum() * 100
    print(f"Variance expliquée par les 2 composantes principales : {variance_explained:.2f}%")

    # Créer une grille pour le tracé de la surface de décision
    x_min, x_max = features_2d[:, 0].min() - 1, features_2d[:, 0].max() + 1
    y_min, y_max = features_2d[:, 1].min() - 1, features_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # PCA inverse transform pour obtenir les caractéristiques originales
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)

    # Faire prédictions sur la grille
    model.head.eval()  # Assurez-vous que le modèle est en mode évaluation
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_points_original, dtype=torch.float32).to(model.head.device)
        outputs = model.head(grid_tensor)
        _, predicted = torch.max(outputs, 1)
        Z = predicted.cpu().numpy().reshape(xx.shape)

    # # Métrics de classification
    from sklearn.metrics import classification_report

    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(targets, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        zero_division=0
    )

    # Tracer la surface de décision
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(
        features_2d[:, 0],
        features_2d[:, 1],
        c=y_true,
        edgecolors='k',
        marker='o'
    )
    plt.title("Decision surface with PCA-reduced features")

    ax = plt.gca()
    fig = plt.gcf()
    present_classes = np.unique(y_true).astype(int)

    if ax.collections:
        handles, _ = ax.collections[-1].legend_elements()
        class_names = [labels[i] for i in present_classes]

        fig.legend(
            handles[:len(class_names)],
            class_names,
            title="Classes",
            loc="upper left",
            bbox_to_anchor=(0.74, 0.45),
            frameon=True
        )

    # Afficher le classification report dans la figure
    fig.text(
        0.75, 0.92, report,
        fontsize=9,
        family="monospace",
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9)
    )

    plt.tight_layout(rect=[0, 0, 0.72, 1])
    plt.show()

def get_model_path(split_strategy, pretrain_dataset, downstream_dataset, backbone, finetune_option, head_type, data_ratio, epochs=50, seed=0):
    """
    Générer le chemin vers le modèle downstream entraîné selon les paramètres.
    """
    assert split_strategy in ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"], f"Invalid split strategy: {split_strategy}"
    assert pretrain_dataset in ["CWRU", "LASPI", "None", "CVRTEST"], f"Invalid pretrain dataset: {pretrain_dataset}"
    assert downstream_dataset in ["CWRU", "LASPI","CVRTEST"], f"Invalid downstream dataset: {downstream_dataset}"
    assert backbone in ["random", "mae", "sap"], f"Invalid backbone type: {backbone}"
    assert head_type in ["linear", "nonlinear"], f"Invalid head type: {head_type}"
    assert 0 < data_ratio <= 1.0, f"Data ratio must be in (0, 1], got {data_ratio}"


    finetune_str = "finetune_True" if finetune_option else "finetune_False"

    if backbone == "random":
        pretrain_dataset = "None"
    
    # Construire le chemin vers le modèle
    model_path = f"results/downstream/{split_strategy}/{pretrain_dataset}_to_{downstream_dataset}_backbone_{backbone}_head_{head_type}_{finetune_str}/data_ratio_{data_ratio}_epochs_{epochs}_seed_{seed}/best_model.pth"
    
    # Vérifier que le chemin existe
    if os.path.exists(model_path) :
        print(f"✓ Model path found: {model_path}")
        return model_path
    else:
        print(f"❌ Model path not found: {model_path}")
        print("Vérifier que les paramètres sont corrects et que le modèle a été entraîné avec ces paramètres!")
        return None

def load_downstream_model(
    model_path: str,
    pretrain_checkpoint_path: str = None,
    classes: np.ndarray = None,
    device: torch.device = "cpu"
):
    """
    Charger le modèle downstream entraîné avec son backbone pré-entraîné.
    
    Supporte les trois types de backbone:
    - MAE: Masked AutoEncoder pré-entraîné
    - SAP: Autres modèles SSL pré-entraînés
    - Random/Scratch: Backbone sans pré-entraînement
    
    Args:
        model_path: Chemin vers le fichier du modèle downstream (best_model.pth ou last_model.pth)
        pretrain_checkpoint_path: Chemin vers le checkpoint de pré-entraînement (optionnel)
        device: Device (cuda ou cpu)
    
    Returns:
        model: Modèle downstream chargé
        config: Configuration du modèle (depuis config.json)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Charger la configuration depuis le répertoire du modèle
    run_dir = os.path.dirname(os.path.dirname(model_path))
    config_path = os.path.join(run_dir, "config.json")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"✓ Configuration chargée depuis: {config_path}")
    print(f"  - Backbone: {config.get('backbone_init', 'unknown')}")
    print(f"  - Pretrain dataset: {config.get('pretrain_dataset', 'N/A')}")
    print(f"  - Downstream dataset: {config['downstream_dataset']}")
    print(f"  - Head type: {config['head_type']}")
    print(f"  - Task: {config['task']}")
    
    # # PARTIE COMMENTÉE - Logique de chargement backbone pré-entraîné (optionnel)
    # # Déterminer le type de backbone
    # backbone_type = config.get('backbone_init', 'mae').lower()
    # 
    # # Toujours créer un backbone random comme base
    # backbone_random = ViT1DEncoder()
    # 
    # # Charger le checkpoint pré-entraîné selon le type
    # if backbone_type == 'random' or config.get('pretrain_dataset') == 'None':
    #     # Pas de pré-entraînement
    #     backbone = backbone_random
    #     print(f"✓ Backbone random (sans pré-entraînement)")
    # 
    # elif backbone_type == 'mae':
    #     # Charger le checkpoint MAE
    #     ssl_mode = MAEModel(
    #         backbone=backbone_random,
    #         args_ssl=SimpleNamespace(mask_ratio=1.0)
    #     )
    #     
    #     if pretrain_checkpoint_path is None:
    #         # Chercher le dernier checkpoint MAE
    #         pretrain_dataset = config.get('pretrain_dataset', 'CWRU')
    #         checkpoints_dir = f"results/pretrain/MAEModel/{pretrain_dataset}Dataset"
    #         
    #         if os.path.exists(checkpoints_dir):
    #             # Récupérer le dernier répertoire de run
    #             run_dirs = sorted([d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))])
    #             if run_dirs:
    #                 latest_run = run_dirs[-1]
    #                 checkpoint_file = os.path.join(checkpoints_dir, latest_run, "checkpoints", "last.pt")
    #                 if os.path.exists(checkpoint_file):
    #                     pretrain_checkpoint_path = checkpoint_file
    #                 else:
    #                     print(f"⚠️  Fichier checkpoint MAE non trouvé: {checkpoint_file}")
    #                     pretrain_checkpoint_path = None
    #             else:
    #                 print(f"⚠️  Aucun répertoire de run trouvé: {checkpoints_dir}")
    #                 pretrain_checkpoint_path = None
    #         else:
    #             print(f"⚠️  Répertoire de checkpoints MAE non trouvé: {checkpoints_dir}")
    #             print("   Utilisation d'un backbone random...")
    #             backbone = backbone_random
    #             pretrain_checkpoint_path = None
    #     
    #     if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
    #         ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path, device=device, strict=False)
    #         backbone = ssl_model.backbone
    #         print(f"✓ Backbone MAE chargé depuis: {pretrain_checkpoint_path}")
    #     else:
    #         backbone = backbone_random
    #         print("⚠️  Backbone random utilisé (aucun checkpoint MAE trouvé)")
    # 
    # elif backbone_type == 'sap':
    #     # Charger le checkpoint SAP
    #     # Note: À adapter selon votre implémentation SAP
    #     try:
    #         from models.ssl.sap import SAPModel
    #         ssl_mode = SAPModel(backbone=backbone_random,
    #                             args_ssl=dict())
    #     except ImportError:
    #         print("⚠️  SAPModel non trouvé, utilisation d'un backbone random")
    #         backbone = backbone_random
    #     else:
    #         if pretrain_checkpoint_path is None:
    #             # Chercher le dernier checkpoint SAP
    #             pretrain_dataset = config.get('pretrain_dataset', 'CWRU')
    #             checkpoints_dir = f"results/pretrain/SAPModel/{pretrain_dataset}Dataset"
    #             
    #             if os.path.exists(checkpoints_dir):
    #                 # Récupérer le dernier répertoire de run
    #                 run_dirs = sorted([d for d in os.listdir(checkpoints_dir) if os.path.isdir(os.path.join(checkpoints_dir, d))])
    #                 if run_dirs:
    #                     latest_run = run_dirs[-1]
    #                     checkpoint_file = os.path.join(checkpoints_dir, latest_run, "checkpoints", "best.pt")
    #                     if os.path.exists(checkpoint_file):
    #                         pretrain_checkpoint_path = checkpoint_file
    #                     else:
    #                         print(f"⚠️  Fichier checkpoint SAP non trouvé: {checkpoint_file}")
    #                         pretrain_checkpoint_path = None
    #                 else:
    #                     print(f"⚠️  Aucun répertoire de run trouvé: {checkpoints_dir}")
    #                     pretrain_checkpoint_path = None
    #             else:
    #                 print(f"⚠️  Répertoire de checkpoints SAP non trouvé: {checkpoints_dir}")
    #                 print("   Utilisation d'un backbone random...")
    #                 backbone = backbone_random
    #                 pretrain_checkpoint_path = None
    #         
    #         if pretrain_checkpoint_path and os.path.exists(pretrain_checkpoint_path):
    #             ssl_model = load_model_checkpoint(ssl_mode, pretrain_checkpoint_path, device=device, strict=False)
    #             backbone = ssl_model.backbone
    #             print(f"✓ Backbone SAP chargé depuis: {pretrain_checkpoint_path}")
    #         else:
    #             backbone = backbone_random
    #             print("⚠️  Backbone random utilisé (aucun checkpoint SAP trouvé)")
    # 
    # else:
    #     print(f"⚠️  Type de backbone inconnu: {backbone_type}, utilisation d'un backbone random")
    #     backbone = backbone_random
    
    # Créer un backbone random (sera remplacé lors du chargement des poids)
    backbone = ViT1DEncoder()
    
    # Reconstruire le modèle downstream avec sa structure
    model = get_downstream_model(
        backbone=backbone,
        task=config['task'],
        head_type=config['head_type'],
        classes=classes if classes is not None else np.array(config.get('classes', [])),
        freeze_backbone=not config.get('finetune', True),
        device=device
    )
    
    # Charger les poids complets du modèle (backbone + head)
    if os.path.exists(model_path):
        model = load_model_checkpoint(model, model_path, device=device)
    else:
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    model.to(device)
    model.eval()
    
    print(f"✓ Modèle complet chargé depuis: {model_path}")
    
    return model, config


def predict(
    model,
    data_loader,
    device: torch.device = "cpu",
    return_features: bool = False,
    return_attention: bool = False,
    return_rollout: bool = False
):
    """
    Faire des prédictions sur un batch de données.
    
    Args:
        model: Modèle entraîné
        data_loader: DataLoader pour les données à prédire
        device: Device (cuda ou cpu)
        return_features: Si True, retourner aussi les features du backbone
        return_attention: Si True, retourner aussi les scores d'attention
        return_rollout: Si True, retourner aussi les scores d'attention rollout (accumulation à travers les couches)
    
    Returns:
        predictions: Prédictions du modèle
        features: Features du backbone (si return_features=True)
        attention_scores: Scores d'attention (si return_attention=True)
        attention_rollout: Scores d'attention rollout (si return_rollout=True)
    """
    
    all_predictions = []
    all_features = []
    all_attention_scores = []
    all_attention_rollout = []
    
    with torch.no_grad():
        for batch in tqdm.tqdm(data_loader, desc="Predicting"):
            # Déplacer le batch sur le device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass avec attention si demandé
            if return_attention:
                outputs, attention_scores = model(batch, get_attention=True)
                all_attention_scores.append(attention_scores.cpu().numpy())
            else:
                outputs = model(batch)
            
            # Récupérer les prédictions
            if model.head.__class__.__name__.endswith("ClassificationHead"):
                # Pour la classification: prendre l'argmax
                # preds = torch.argmax(outputs, dim=1)
                preds = outputs
                probs = torch.softmax(outputs, dim=1)
                targets = model.head.transform_labels(batch['y_label'])
                all_predictions.append({
                    'logits': outputs.cpu().numpy(),
                    'predictions': preds.cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                    'probabilities': probs.cpu().numpy(),
                })
            else:
                # Pour la régression
                targets = batch['y_label']
                all_predictions.append({
                    'predictions': outputs.cpu().numpy(),
                    'targets': targets.cpu().numpy(),
                })
            
            # Features du backbone (si demandé)
            if return_features:
                x_raw = batch['X_raw']
                x_norm = normalization.global_z_log_normalization(
                    x=x_raw, 
                    stats=model.backbone.stats
                )
                features = model.backbone(x_norm)
                cls_token = features[:, 0]  # CLS token
                all_features.append(cls_token.cpu().numpy())
            
            # Attention rollout (si demandé)
            if return_rollout:
                x_raw = batch['X_raw']
                x_norm = normalization.global_z_log_normalization(
                    x=x_raw, 
                    stats=model.backbone.stats
                )
                rollout_scores = model.backbone.get_attention_rollout(x_norm)
                all_attention_rollout.append(rollout_scores.cpu().numpy())
    
    # Concaténer les résultats
    predictions = {
        k: np.concatenate([d[k] for d in all_predictions])
        for k in all_predictions[0].keys()
    }
    
    features = np.concatenate(all_features) if all_features else None
    attention_scores = np.concatenate(all_attention_scores, axis=0) if all_attention_scores else None
    attention_rollout = np.concatenate(all_attention_rollout, axis=0) if all_attention_rollout else None

    # Normalize attention scores
    if attention_scores is not None:
        print("Attention scores shape:", attention_scores.shape)
        attention_scores = attention_scores / (attention_scores.sum(axis=-1, keepdims=True) + 1e-8)
    
    # Normalize attention rollout
    if attention_rollout is not None:
        print("Attention rollout shape:", attention_rollout.shape)
        attention_rollout = attention_rollout / (attention_rollout.sum(axis=-1, keepdims=True) + 1e-8)
    
    return predictions, features, attention_scores, attention_rollout


def infer_single_sample(model, sample_data: torch.Tensor, device: torch.device = "cpu"):
    """
    Faire une prédiction sur un seul échantillon.
    
    Args:
        model: Modèle entraîné
        sample_data: Tensor de shape (1, window_size) ou dict avec clé 'X_raw'
        device: Device (cuda ou cpu)
    
    Returns:
        Prédiction pour cet échantillon
    """
    
    model.eval()
    
    with torch.no_grad():
        # Si c'est un tensor, créer un batch
        if isinstance(sample_data, torch.Tensor):
            batch = {'X_raw': sample_data.unsqueeze(0).to(device) if sample_data.dim() == 1 
                    else sample_data.to(device)}
        else:
            batch = sample_data
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
        
        outputs = model(batch)
        
        if model.head.__class__.__name__.endswith("ClassificationHead"):
            pred = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)
            class_name = model.head.lb.classes_[pred.item()]
            return {
                'predicted_class_idx': pred.item(),
                'predicted_class': class_name,
                'logits': outputs.cpu().numpy(),
                'probabilities': prob.cpu().numpy(),
                'confidence': prob.max().item()
            }
        else:
            return {
                'prediction': outputs.cpu().numpy().flatten()
            }

if __name__ == "__main__":
    # debug

    from types import SimpleNamespace
    from dataset import split_data_factory

    # Dataloader configuration
    args_dataloader = SimpleNamespace(
        name="LASPI", #downstream dataset name
        window_size=2048,
        window_stride=512,
        batch_size=256,
        data_ratio=0.01,
        seed=0,
    )

    # Split dataloaders
    train_loader, valid_loader, test_loader, labels, dataset = split_data_factory.split_dataloader(
        split_type="speed_load_stratified", #split type
        args_dataloader=args_dataloader,
    )   

    # Get model path
    model_path = get_model_path(
        split_strategy="speed_load_stratified",
        pretrain_dataset="CWRU",
        downstream_dataset="LASPI",
        backbone="sap",
        finetune_option=False,
        head_type="linear",
        data_ratio=0.01,
        epochs=100,
    )

    # Charger le modèle
    model, config = load_downstream_model(
        model_path=model_path,
        classes=labels,
        device="cuda"
    )

    # Prédictions
    predictions, features, attention_scores = predict(
        model, 
        test_loader,
        device="cuda",
        return_features=True,
        return_attention=True
    )