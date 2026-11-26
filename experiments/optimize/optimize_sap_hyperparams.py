"""
Script d'optimisation des hyperparamètres du SAP avec Optuna.
Optimise les hyperparamètres du préentraînement SAP (Spectral Aliasing Pretext)
avec architecture et preprocessing fixés.
Retourne les meilleurs hyperparamètres trouvés.

Architecture fixée (ViT Base):
- hidden_dim: 512
- n_layers: 8
- heads: 8

Préprocessing fixé:
- downsampling_factor: 2
- loss_domain: 'linear'
- symmetric_spectrum: True
- revin: True

Hyperparamètres optimisés (3 seulement):
- learning_rate: Taux d'apprentissage
- weight_decay: Régularisation L2
- batch_size: Taille du batch

Utilisation:
    python experiments/optimize/optimize_sap_hyperparams.py
    python experiments/optimize/optimize_sap_hyperparams.py --n_trials 100
    python experiments/optimize/optimize_sap_hyperparams.py --n_trials 50 --dataset LASPI
"""

import json
import os
import optuna
import subprocess
import csv
import argparse
from pathlib import Path
from datetime import datetime
from optuna.trial import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner


# Configuration par défaut
CONFIG = {
    "pretrain_dataset": ["PHM07", "PHM09", "PHM12", "CWRU", "LASPI", "HUST", "PADERBORN"],  # Multi-dataset
    "optimization_runs": 30,  # Nombre de trials Optuna
    "epochs_per_trial": 20,   # Epochs d'entraînement par trial (réduit pour optimisation)
    # Architecture ViT Base (fixée)
    "hidden_dim": 512,
    "n_layers": 8,
    "heads": 8,
    # Préprocessing et spectral (fixés)
    "downsampling_factor": 2,
    "loss_domain": "linear",
    "symmetric_spectrum": True,
    "revin": True,
}


def build_command(trial: Trial, config: dict) -> str:
    """
    Construit la commande d'entraînement SAP avec hyperparamètres suggérés.
    
    Args:
        trial: Trial Optuna
        config: Configuration globale
        
    Returns:
        str: Commande à exécuter
    """
    
    # Suggérer des hyperparamètres avec des ranges pertinents
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-7, 1e-4, log=True)
    
    # Architecture ViT Base (fixée)
    hidden_dim = config['hidden_dim']
    n_layers = config['n_layers']
    heads = config['heads']
    
    # Paramètres fixés (ne pas les suggérer, utiliser directement)
    downsampling_factor = config['downsampling_factor']
    loss_domain = config['loss_domain']
    symmetric_spectrum = config['symmetric_spectrum']
    revin = config['revin']
    
    # Afficher les hyperparamètres
    print(f"\n{'='*70}")
    print(f"TRIAL {trial.number} - Hyperparamètres suggérés")
    print(f"{'='*70}")
    print(f"batch_size:           {batch_size}")
    print(f"learning_rate:        {learning_rate:.2e}")
    print(f"weight_decay:         {weight_decay:.2e}")
    print(f"downsampling_factor:  {downsampling_factor}")
    print(f"loss_domain:          {loss_domain}")
    print(f"symmetric_spectrum:   {symmetric_spectrum}")
    print(f"revin:                {revin}")
    print(f"\n--- Architecture ViT Base (fixée) ---")
    print(f"hidden_dim:           {hidden_dim}")
    print(f"n_layers:             {n_layers}")
    print(f"heads:                {heads}")
    print(f"epochs:               {config['epochs_per_trial']}")
    print(f"{'='*70}\n")
    
    # Construire la commande
    cmd = (
        f"python experiments/pretrain/pretrain_sap.py "
        f"--pretrain_dataset {' '.join(config['pretrain_dataset'])} "
        f"--batch_size {batch_size} "
        f"--learning_rate {learning_rate} "
        f"--weight_decay {weight_decay} "
        f"--hidden_dim {hidden_dim} "
        f"--n_layers {n_layers} "
        f"--heads {heads} "
        f"--downsampling_factor {downsampling_factor} "
        f"--loss_domain {loss_domain} "
        f"--epochs {config['epochs_per_trial']} "
        f"--window_size 2048 "
        f"--window_stride 256 "
    )
    
    # Ajouter les flags booléens
    if symmetric_spectrum:
        cmd += "--symetric_spectrum "  # Note: typo dans le code original
    if revin:
        cmd += "--revin "
    
    return cmd


def objective(trial: Trial, config: dict) -> float:
    """
    Fonction objectif pour Optuna.
    Suggère des hyperparamètres, lance l'entraînement SAP et retourne la validation loss.
    
    Args:
        trial: Trial Optuna courant
        config: Configuration globale
        
    Returns:
        float: Validation loss à minimiser
    """
    
    # Construire la commande
    cmd = build_command(trial, config)
    
    print(f"\n📊 Exécution de l'entraînement SAP...")
    print(f"Commande: {cmd}\n")
    
    try:
        # Exécuter l'entraînement avec timeout
        result = subprocess.run(
            cmd,
            shell=True,
            text=True,
            timeout=7200,  # 2 heures timeout
            capture_output=True
        )
        
        if result.returncode != 0:
            print(f"❌ Erreur lors de l'exécution:")
            print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return float('inf')
        
        # Extraire les métriques d'entraînement pour le pruning
        metrics = extract_training_metrics(config)
        
        if metrics is None:
            print("❌ Impossible d'extraire les métriques")
            return float('inf')
        
        # Reporter les métriques intermédiaires au pruner
        # Le pruner vérifiera après chaque epoch si le trial doit être prunné
        for epoch, loss in enumerate(metrics['validation_losses']):
            trial.report(loss, epoch)
            
            # Vérifier si le trial doit être prunné
            if trial.should_prune():
                print(f"\n⏸️  Trial prunné à l'epoch {epoch}")
                raise optuna.TrialPruned()
        
        # Dernière validation loss
        validation_loss = metrics['validation_losses'][-1]
        
        print(f"✓ Validation Loss final: {validation_loss:.6f}\n")
        return validation_loss
        
    except optuna.TrialPruned:
        # Le trial a été prunné (c'est normal)
        print(f"✓ Trial {trial.number} prunné (perf médiane insuffisante)\n")
        raise
    except subprocess.TimeoutExpired:
        print("❌ Timeout: Entraînement trop long (> 2 heures)")
        return float('inf')
    except Exception as e:
        print(f"❌ Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return float('inf')


def extract_training_metrics(config: dict) -> dict:
    """
    Extrait les métriques d'entraînement (courbes d'epochs) du fichier de logs.
    Utilisé pour le pruning avec MedianPruner.
    
    Args:
        config: Configuration globale (contient le dataset)
        
    Returns:
        dict: Dictionnaire contenant:
            - 'validation_losses': liste des validation loss par epoch
            - 'training_losses': liste des training loss par epoch
    """
    try:
        # Trouver le répertoire de résultats le plus récent
        pretrain_dir = Path("results/pretrain/SAPModel")
        
        if not pretrain_dir.exists():
            print(f"⚠️  Répertoire non trouvé: {pretrain_dir}")
            return None
        
        # Si dataset est une liste, prendre le premier (ou ConcatDataset si multi)
        dataset_name = config['pretrain_dataset']
        if isinstance(dataset_name, list):
            dataset_dir = "ConcatDataset"
        else:
            dataset_dir = f"{dataset_name}Dataset"
        
        # Chercher tous les répertoires de résultats
        result_dirs = list((pretrain_dir / dataset_dir).glob("*/"))
        
        if not result_dirs:
            print(f"⚠️  Pas de répertoires trouvés dans: {pretrain_dir / dataset_dir}")
            return None
        
        # Prendre le plus récent
        latest_dir = max(result_dirs, key=lambda p: p.stat().st_mtime)
        
        # Chercher le fichier de logs.json
        metrics_json = latest_dir / "logs.json"
        
        if metrics_json.exists():
            print(f"📄 Lecture des métriques de pruning: {metrics_json}")
            with open(metrics_json, 'r') as f:
                logs = json.load(f)
            
            # Extraire les listes de losses
            validation_losses = []
            training_losses = []
            
            if isinstance(logs, dict):
                # Format dict avec listes
                if 'valid_loss' in logs:
                    validation_losses = logs['valid_loss']
                elif 'validation_loss' in logs:
                    validation_losses = logs['validation_loss']
                
                if 'train_loss' in logs:
                    training_losses = logs['train_loss']
                elif 'training_loss' in logs:
                    training_losses = logs['training_loss']
            
            elif isinstance(logs, list):
                # Format liste d'epochs
                for entry in logs:
                    if 'valid_loss' in entry:
                        validation_losses.append(entry['valid_loss'])
                    elif 'validation_loss' in entry:
                        validation_losses.append(entry['validation_loss'])
                    
                    if 'train_loss' in entry:
                        training_losses.append(entry['train_loss'])
                    elif 'training_loss' in entry:
                        training_losses.append(entry['training_loss'])
            
            if validation_losses:
                return {
                    'validation_losses': validation_losses,
                    'training_losses': training_losses
                }
        
        print(f"⚠️  Pas de fichier logs.json trouvé")
        return None
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction des métriques: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def extract_validation_loss(config: dict) -> float:
    """
    Extrait la dernière validation loss du fichier de métriques d'entraînement.
    
    Args:
        config: Configuration globale (contient le dataset)
        
    Returns:
        float: Dernière validation loss trouvée, ou None si impossible
    """
    try:
        # Trouver le répertoire de résultats le plus récent
        pretrain_dir = Path("results/pretrain/SAPModel")
        
        if not pretrain_dir.exists():
            print(f"⚠️  Répertoire non trouvé: {pretrain_dir}")
            return None
        
        # Si dataset est une liste, prendre le premier (ou ConcatDataset si multi)
        dataset_name = config['pretrain_dataset']
        if isinstance(dataset_name, list):
            dataset_dir = "ConcatDataset"
        else:
            dataset_dir = f"{dataset_name}Dataset"
        
        # Chercher tous les répertoires de résultats
        result_dirs = list((pretrain_dir / dataset_dir).glob("*/"))
        
        if not result_dirs:
            print(f"⚠️  Pas de répertoires de résultats trouvés dans: {pretrain_dir / dataset_dir}")
            return None
        
        # Prendre le plus récent (dernier créé)
        latest_dir = max(result_dirs, key=lambda p: p.stat().st_mtime)
        
        # Chercher le fichier de métriques (logs.json ou metrics.csv)
        metrics_json = latest_dir / "logs.json"
        metrics_csv = latest_dir / "metrics.csv"
        
        # Préférer JSON (format plus standard)
        if metrics_json.exists():
            print(f"📄 Lecture des métriques: {metrics_json}")
            with open(metrics_json, 'r') as f:
                logs = json.load(f)
                
            # Chercher la dernière validation loss
            if isinstance(logs, dict):
                if 'valid_loss' in logs and len(logs['valid_loss']) > 0:
                    return float(logs['valid_loss'][-1])
                elif 'validation_loss' in logs and len(logs['validation_loss']) > 0:
                    return float(logs['validation_loss'][-1])
            elif isinstance(logs, list) and len(logs) > 0:
                last_entry = logs[-1]
                if 'valid_loss' in last_entry:
                    return float(last_entry['valid_loss'])
                elif 'validation_loss' in last_entry:
                    return float(last_entry['validation_loss'])
        
        # Fallback: lire CSV
        elif metrics_csv.exists():
            print(f"📄 Lecture des métriques: {metrics_csv}")
            with open(metrics_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if rows:
                    last_row = rows[-1]
                    if 'valid_loss' in last_row:
                        return float(last_row['valid_loss'])
                    elif 'validation_loss' in last_row:
                        return float(last_row['validation_loss'])
        
        print(f"⚠️  Pas de fichier de métriques trouvé dans: {latest_dir}")
        return None
        
    except json.JSONDecodeError as e:
        print(f"❌ Erreur JSON: {str(e)}")
        return None
    except Exception as e:
        print(f"❌ Erreur lors de l'extraction: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_results(study: optuna.Study, config: dict):
    """
    Sauvegarde les résultats de l'optimisation dans un fichier JSON.
    
    Args:
        study: Étude Optuna complétée
        config: Configuration utilisée
    """
    
    # Créer le répertoire de résultats
    opt_dir = Path("results/optimization")
    opt_dir.mkdir(parents=True, exist_ok=True)
    
    # Générer un nom unique avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = opt_dir / f"sap_optimization_{timestamp}.json"
    
    # Préparer les résultats
    results = {
        "timestamp": timestamp,
        "method": "SAP",
        "dataset": config['pretrain_dataset'] if isinstance(config['pretrain_dataset'], str) else "ConcatDataset",
        "n_trials": len(study.trials),
        "best_trial": {
            "number": study.best_trial.number,
            "value": float(study.best_trial.value),
            "params": study.best_trial.params,
        },
        "trials": []
    }
    
    # Ajouter tous les trials
    for trial in study.trials:
        results["trials"].append({
            "number": trial.number,
            "state": str(trial.state),
            "value": float(trial.value) if trial.value is not None else None,
            "params": trial.params,
        })
    
    # Sauvegarder en JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Résultats sauvegardés: {results_file}")
    
    return results_file


def print_summary(study: optuna.Study):
    """Affiche un résumé des résultats d'optimisation."""
    
    print(f"\n{'='*70}")
    print(f"RÉSUMÉ DE L'OPTIMISATION OPTUNA - SAP")
    print(f"{'='*70}\n")
    
    # Meilleur trial
    print(f"🏆 MEILLEUR TRIAL")
    print(f"   Numéro: {study.best_trial.number}")
    print(f"   Validation Loss: {study.best_trial.value:.6f}")
    print(f"\n   Hyperparamètres:")
    for key, value in sorted(study.best_trial.params.items()):
        print(f"      {key:25s}: {value}")
    
    # Statistiques
    print(f"\n📊 STATISTIQUES")
    print(f"   Nombre de trials: {len(study.trials)}")
    print(f"   Nombre de trials réussis: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)}")
    print(f"   Nombre de trials prunés: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)}")
    print(f"   Nombre de trials échoués: {sum(1 for t in study.trials if t.state == optuna.trial.TrialState.FAIL)}")
    
    # Trials les plus proches du meilleur
    print(f"\n🔝 TOP 5 TRIALS")
    best_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))[:5]
    for i, trial in enumerate(best_trials, 1):
        print(f"   {i}. Trial {trial.number}: Loss = {trial.value:.6f}")
    
    print(f"\n{'='*70}\n")


def run_optimization(config: dict, seed: int = 42):
    """
    Lance l'optimisation Optuna pour SAP.
    
    Args:
        config: Configuration globale
        seed: Seed pour le TPE sampler
    """
    
    print(f"\n{'='*70}")
    print(f"OPTIMISATION OPTUNA - SAP (Spectral Aliasing Pretext)")
    print(f"{'='*70}")
    print(f"\n🏗️  Architecture ViT Base (fixée):")
    print(f"  hidden_dim: {config['hidden_dim']}")
    print(f"  n_layers:   {config['n_layers']}")
    print(f"  heads:      {config['heads']}")
    print(f"\n📊 Configuration d'optimisation:")
    
    # Afficher les datasets
    if isinstance(config['pretrain_dataset'], list):
        dataset_str = " + ".join(config['pretrain_dataset'])
        print(f"  Dataset (Multi): {dataset_str}")
    else:
        print(f"  Dataset: {config['pretrain_dataset']}")
    
    print(f"  Nombre de trials: {config['optimization_runs']}")
    print(f"  Epochs par trial: {config['epochs_per_trial']}")
    print(f"  Seed Optuna: {seed}")
    print(f"\n🔍 Hyperparamètres à optimiser:")
    print(f"  - learning_rate")
    print(f"  - weight_decay")
    print(f"  - batch_size")
    print(f"  - downsampling_factor")
    print(f"  - loss_domain")
    print(f"  - symmetric_spectrum")
    print(f"  - revin")
    print(f"\n{'='*70}\n")
    
    # Créer une étude Optuna avec TPE sampler et MedianPruner
    sampler = TPESampler(seed=seed)
    pruner = MedianPruner(
        n_startup_trials=5,      # Laisser 5 trials se finir avant de pruner
        n_warmup_steps=5,        # Warmup de 5 steps avant de commencer le pruning
        interval_steps=1         # Vérifier à chaque step
    )
    
    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        study_name="SAP_Hyperparameter_Optimization"
    )
    
    # Lancer l'optimisation
    try:
        study.optimize(
            lambda trial: objective(trial, config),
            n_trials=config['optimization_runs'],
            show_progress_bar=True
        )
        
        # Afficher le résumé
        print_summary(study)
        
        # Sauvegarder les résultats
        results_file = save_results(study, config)
        
        # Générer la commande optimale
        print(f"\n💡 COMMANDE OPTIMALE (ViT Base)")
        print(f"{"="*70}")
        best_params = study.best_trial.params
        
        cmd = (
            f"python submit/submit_pretrain.py sap "
            f"--batch_size {best_params.get('batch_size', 32)} "
            f"--learning_rate {best_params.get('learning_rate', 8.3e-4)} "
            f"--weight_decay {best_params.get('weight_decay', 3.6e-06)} "
            f"--hidden_dim {config['hidden_dim']} "
            f"--n_layers {config['n_layers']} "
            f"--heads {config['heads']} "
            f"--downsampling_factor {best_params.get('downsampling_factor', 2)} "
            f"--loss_domain {best_params.get('loss_domain', 'log')} "
        )
        
        if best_params.get('symmetric_spectrum', True):
            cmd += "--symetric_spectrum "
        if best_params.get('revin', True):
            cmd += "--revin "
        
        print(f"\n{cmd}\n")
        print(f"{"="*70}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Optimisation interrompue par l'utilisateur")
        print_summary(study)
        return 1
    except Exception as e:
        print(f"\n❌ Erreur lors de l'optimisation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Optimisation des hyperparamètres SAP avec Optuna"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Nombre de trials Optuna à exécuter (default: 50)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset de préentraînement (default: multi-dataset ConcatDataset)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Nombre d'epochs par trial (default: 20)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour Optuna TPE sampler (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Mettre à jour la configuration
    CONFIG['optimization_runs'] = args.n_trials
    
    # Gérer le dataset (par défaut multi-dataset, ou utiliser --dataset pour single)
    if args.dataset:
        CONFIG['pretrain_dataset'] = args.dataset
    # else: garder la liste multi-dataset par défaut
    
    CONFIG['epochs_per_trial'] = args.epochs
    
    # Set PYTHONPATH to include current directory
    pythonpath = os.getenv("PYTHONPATH", "")
    if pythonpath:
        pythonpath = pythonpath + ":" + os.path.abspath(".")
    else:
        pythonpath = os.path.abspath(".")
    os.environ["PYTHONPATH"] = pythonpath
    
    # Lancer l'optimisation
    return run_optimization(CONFIG, seed=args.seed)


if __name__ == "__main__":
    exit(main())
