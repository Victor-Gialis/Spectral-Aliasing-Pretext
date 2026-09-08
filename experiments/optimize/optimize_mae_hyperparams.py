"""
Script simple d'optimisation des hyperparamètres du MAE avec Optuna.
Optimise et retourne les meilleurs hyperparamètres.
"""

import json
import os
import optuna
import subprocess
from pathlib import Path
from optuna.trial import Trial
from optuna.samplers import TPESampler


# Configuration
CONFIG = {
    "pretrain_dataset": "CWRU",
    "optimization_runs": 50,  # Nombre de trials Optuna
}


def objective(trial: Trial) -> float:
    """
    Fonction objectif pour Optuna.
    Suggère des hyperparamètres et lance l'entraînement du MAE.
    Retourne la validation loss comme métrique à minimiser.
    """
    
    # Suggérer des hyperparamètres
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256, 512])
    mask_ratio = trial.suggest_float('mask_ratio', 0.25, 0.75, step=0.05)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    epochs = 30
    
    print(f"\n{'='*60}")
    print(f"Trial {trial.number}")
    print(f"{'='*60}")
    print(f"batch_size: {batch_size}")
    print(f"mask_ratio: {mask_ratio:.3f}")
    print(f"learning_rate: {learning_rate:.2e}")
    print(f"weight_decay: {weight_decay:.2e}")
    print(f"epochs: {epochs}")
    
    # Construire la commande d'entraînement
    cmd = (
        f"python experiments/pretrain_mae.py "
        f"--pretrain_dataset {CONFIG['pretrain_dataset']} "
        f"--batch_size {batch_size} "
        f"--mask_ratio {mask_ratio} "
        f"--learning_rate {learning_rate} "
        f"--weight_decay {weight_decay} "
        f"--epochs {epochs} "
        f"--window_size 2048 "
        f"--window_stride 256 "
    )
    
    try:
        
        # Exécuter l'entraînement
        result = subprocess.run(cmd, shell=True,  text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"❌ Erreur lors de l'exécution")
            return float('inf')
        
        # Extraire la validation loss
        validation_loss = extract_validation_loss()
        
        if validation_loss is None:
            print("❌ Impossible d'extraire la validation loss")
            return float('inf')
        
        print(f"✓ Validation Loss: {validation_loss:.6f}\n")
        return validation_loss
        
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Entraînement trop long")
        return float('inf')
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        return float('inf')


def extract_validation_loss() -> float:
    """
    Extrait la dernière validation loss du fichier de métriques.
    Assume que les résultats sont stockés dans results/pretrain/.
    
    Returns:
        float: Dernière validation loss ou None si non trouvée
    """
    try:
        # Trouver le répertoire de résultats le plus récent
        pretrain_dir = Path("results/pretrain")
        if not pretrain_dir.exists():
            return None
        
        # Chercher dans tous les répertoires MAE/CWRU
        mae_dirs = list((pretrain_dir / "MAEModel" / (CONFIG['pretrain_dataset']+'Dataset')).glob("*/log/metrics.csv"))
        
        if not mae_dirs:
            return None
        
        # Prendre le plus récent
        latest_metrics = max(mae_dirs, key=lambda p: p.stat().st_mtime)
        
        # Lire la dernière ligne du fichier CSV
        with open(latest_metrics, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                # La dernière ligne contient les métriques du dernier epoch
                last_line = lines[-1].strip()
                # Format: epoch,train_loss,valid_loss
                parts = last_line.split(',')
                if len(parts) >= 3:
                    return float(parts[2])
        
        return None
        
    except Exception as e:
        print(f"Erreur lors de l'extraction de la validation loss: {str(e)}")
        return None


def run_optimization():
    """Lance l'optimisation Optuna simple et affiche les résultats."""
    
    print(f"\n{'='*60}")
    print(f"OPTIMISATION OPTUNA - MAE")
    print(f"{'='*60}\n")
    
    # Créer une étude Optuna
    sampler = TPESampler(seed=42)
    
    study = optuna.create_study(
        sampler=sampler,
        direction="minimize",
    )
    
    # Lancer l'optimisation
    try:
        study.optimize(
            objective,
            n_trials=CONFIG['optimization_runs'],
            n_jobs=1,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Optimisation interrompue par l'utilisateur")
    
    # Afficher les résultats
    print_results(study)

    # Sauvegarder les meilleurs hyperparamètres
    save_results(study)


def print_results(study: optuna.Study):
    """Affiche les meilleurs hyperparamètres trouvés."""
    
    print(f"\n{'='*60}")
    print(f"RÉSULTATS")
    print(f"{'='*60}\n")
    
    best_trial = study.best_trial
    
    print(f"✓ Meilleure Validation Loss: {best_trial.value:.6f}\n")
    print("Meilleurs Hyperparamètres:")
    print("-" * 60)
    for key, value in best_trial.params.items():
        print(f"  {key:20s}: {value}")
    print("-" * 60)

def save_results(study: optuna.Study):
    """Sauvegarde les meilleurs hyperparamètres dans un fichier JSON."""
    
    output_dir = Path("results/mae_optimization")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_trial = study.best_trial

    timestamp = best_trial.datetime_start.strftime("%Y%m%d_%H%M%S")
    config_file = output_dir / f"best_hyperparams_{timestamp}.json"
        
    best_trial = study.best_trial
    with open(config_file, "w") as f:
        json.dump({
            "validation_loss": float(best_trial.value),
            "hyperparameters": best_trial.params
        }, f, indent=2)
    
    print(f"✓ Configuration sauvegardée dans {config_file}")

if __name__ == "__main__":
    # Set PYTHONPATH to include current directory
    pythonpath = os.getenv("PYTHONPATH", "")
    if pythonpath:
        pythonpath = pythonpath + ":" + os.path.abspath(".")
    else:
        pythonpath = os.path.abspath(".")
    os.environ["PYTHONPATH"] = pythonpath
    
    # Lancer l'optimisation
    run_optimization()
