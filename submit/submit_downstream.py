import os
import argparse
import subprocess
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Common parameters (default values)
COMMON_PARAMS = {
    "downstream_dataset": "PADERBORN",
    "data_ratios": [1.0],
    "finetune_options": [False, True],
    "head_type": "linear",
    "batch_size": 64,
    "window_size": 2048,
    "window_stride": 2048,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 50,
    "task": "classification",
    "seeds": [0],
}

# Backbone-specific configurations
BACKBONE_CONFIGS = {
    "random": {
        "checkpoint_path": None,
        "pretrain_dataset": "CWRU",
    },
    "mae": {
        "checkpoint_path": "results/pretrain/MAEModel/ConcatDataset/20261102_153220",
        "pretrain_dataset": "Concat",
    },
    "sap": {
        "checkpoint_path": "results/pretrain/SAPModel/ConcatDataset/20261116_164823",
        "pretrain_dataset": "Concat",
    },
    "sajepa": {
        "checkpoint_path": "results/pretrain/SAJEPAModel/CWRUDataset/20261105_105419",
        "pretrain_dataset": "CWRU",
    },
    "ijepa": {
        "checkpoint_path": None,
        "pretrain_dataset": "CWRU",
    },
    "ts2vec": {
        "checkpoint_path": None,
        "pretrain_dataset": "CWRU",
    },
    "tstcc": {
        "checkpoint_path": None,
        "pretrain_dataset": "CWRU",
    },
}

# Map downstream dataset to split type
SPLIT_TYPE_MAP = {
    "CWRU": "speed_stratified",
    "LASPI": "speed_load_stratified",
    "PADERBORN": "speed_stratified",
}


def get_split_type(downstream_dataset):
    """Get split type based on downstream dataset."""
    return SPLIT_TYPE_MAP.get(downstream_dataset, "speed_stratified")


def get_epochs_for_ratio(data_ratio):
    """Determine epochs based on data ratio."""
    if data_ratio == 0.01:
        return 100
    elif data_ratio == 0.05:
        return 50
    else:
        return 50


def merge_configs(common_params_override=None):
    """Merge common params with CLI overrides."""
    config = COMMON_PARAMS.copy()
    
    if common_params_override:
        config.update(common_params_override)
    
    return config


def build_command(backbone, config):
    """Build the downstream command for a given backbone and config."""
    backbone_config = BACKBONE_CONFIGS.get(backbone, {})
    checkpoint_path = backbone_config.get("checkpoint_path")
    pretrain_dataset = backbone_config.get("pretrain_dataset", "CWRU")
    
    cmd = f"python experiments/downstream/downstream_{backbone}.py"
    
    # Add parameters
    cmd += f" --pretrain_dataset {pretrain_dataset}"
    cmd += f" --downstream_dataset {config['downstream_dataset']}"
    cmd += f" --batch_size {config['batch_size']}"
    cmd += f" --window_size {config['window_size']}"
    cmd += f" --window_stride {config['window_stride']}"
    cmd += f" --learning_rate {config['learning_rate']}"
    cmd += f" --weight_decay {config['weight_decay']}"
    cmd += f" --head_type {config['head_type']}"
    cmd += f" --task {config['task']}"
    cmd += f" --split_type {get_split_type(config['downstream_dataset'])}"
    
    if checkpoint_path:
        cmd += f" --backbone_checkpoint_path {checkpoint_path}"
    
    return cmd


def run_downstream_experiments(backbones, config, args):
    """Run downstream experiments for all combinations."""
    data_ratios = config["data_ratios"]
    finetune_options = config["finetune_options"]
    seeds = config["seeds"]
    
    total_jobs = len(backbones) * len(data_ratios) * len(finetune_options) * len(seeds)
    current_job = 0
    
    for backbone, data_ratio, finetune, seed in product(backbones, data_ratios, finetune_options, seeds):
        current_job += 1
        
        # Skip random backbone without finetuning
        if not finetune and backbone == "random":
            print(f"[{current_job}/{total_jobs}] Skipping random backbone without finetuning")
            continue
        
        # Determine epochs based on data ratio
        epochs = get_epochs_for_ratio(data_ratio)
        
        # Build command
        cmd = build_command(backbone, config)
        cmd += f" --data_ratio {data_ratio}"
        cmd += f" --epochs {epochs}"
        cmd += f" --seed {seed}"
        
        if finetune:
            cmd += " --finetune"
        
        print(f"\n[{current_job}/{total_jobs}] Starting downstream evaluation with {backbone.upper()} backbone...")
        print(f"Running: {cmd}")
        
        if args.mode == "jean-zay":
            submit_to_jean_zay(backbone, cmd, args, data_ratio, finetune, seed)
        else:
            exit_code = os.system(cmd)
            if exit_code != 0:
                print(f"❌ Command failed: {cmd}")


def submit_to_jean_zay(backbone, cmd, args, data_ratio, finetune, seed):
    """Generate and optionally submit SLURM script for Jean-Zay."""
    finetune_str = "ft" if finetune else "noft"
    job_name = f"{backbone}_downstream_ratio{data_ratio}_{finetune_str}_seed{seed}"
    script_name = f"slurm_{job_name}.sh"
    
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={args.time}
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.nodes}
#SBATCH --gpus-per-node={args.gpus_per_node}
#SBATCH --cpus-per-task={args.cpus_per_task}
#SBATCH --output={job_name}_%j.log
#SBATCH --error={job_name}_%j.err

# Load modules
module purge
module load pytorch-gpu/py3/2.0.1

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run the training command
{cmd}
"""
    
    # Write script to file
    with open(script_name, "w") as f:
        f.write(slurm_script)
    
    os.chmod(script_name, 0o755)
    print(f"✓ SLURM script generated: {script_name}")
    
    if args.submit:
        print(f"📤 Submitting job to Jean-Zay...")
        result = subprocess.run(["sbatch", script_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Job submitted!")
            print(result.stdout)
        else:
            print(f"❌ Job submission failed!")
            print(result.stderr)
    elif not args.script_only:
        print(f"To submit: sbatch {script_name}")


def main():
    parser = argparse.ArgumentParser(
        description="Run downstream experiments for different backbone initializations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python submit_downstream.py mae
  python submit_downstream.py sap --downstream_dataset CWRU --epochs 30
  python submit_downstream.py mae ijepa --data_ratios 0.1 0.2 --seeds 0 1 2
  python submit_downstream.py sap --mode jean-zay --submit --time 10:00:00
        """
    )
    
    parser.add_argument(
        "backbones",
        nargs="+",
        choices=list(BACKBONE_CONFIGS.keys()),
        help="Backbone initialization method(s) to evaluate"
    )
    
    # Mode: local or jean-zay
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "jean-zay"],
        default="local",
        help="Execution mode: local or submit to Jean-Zay cluster"
    )
    
    # SLURM/Jean-Zay specific options
    parser.add_argument(
        "--time",
        type=str,
        default="20:00:00",
        help="SLURM time limit (default: 20:00:00)"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu_p100",
        help="SLURM partition (default: gpu_p100)"
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Number of nodes (default: 1)"
    )
    parser.add_argument(
        "--gpus_per_node",
        type=int,
        default=1,
        help="GPUs per node (default: 1)"
    )
    parser.add_argument(
        "--cpus_per_task",
        type=int,
        default=10,
        help="CPUs per task (default: 10)"
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit job to SLURM (jean-zay mode only)"
    )
    parser.add_argument(
        "--script_only",
        action="store_true",
        help="Generate SLURM scripts without submitting (jean-zay mode only)"
    )
    
    # Optional arguments to override common parameters
    parser.add_argument(
        "--downstream_dataset",
        type=str,
        help="Override downstream dataset"
    )
    parser.add_argument(
        "--data_ratios",
        type=float,
        nargs="+",
        help="Override data ratios"
    )
    parser.add_argument(
        "--finetune_options",
        type=lambda x: [y.lower() == "true" for y in x.split(",")],
        help="Override finetune options (e.g., 'true,false')"
    )
    parser.add_argument(
        "--head_type",
        type=str,
        help="Override head type (linear or non-linear)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override batch size"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Override window size"
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        help="Override window stride"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Override weight decay"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override epochs"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override seeds"
    )
    
    args = parser.parse_args()
    
    # Collect CLI overrides for common parameters
    common_overrides = {}
    for param in ["downstream_dataset", "data_ratios", "finetune_options", "head_type",
                  "batch_size", "window_size", "window_stride", "learning_rate", 
                  "weight_decay", "epochs", "seeds"]:
        value = getattr(args, param, None)
        if value is not None:
            common_overrides[param] = value
    
    # Merge configs
    config = merge_configs(common_overrides)
    
    # Run experiments
    run_downstream_experiments(args.backbones, config, args)
    
    print(f"\n✓ All downstream experiments completed!")
    return 0


if __name__ == "__main__":
    exit(main())
