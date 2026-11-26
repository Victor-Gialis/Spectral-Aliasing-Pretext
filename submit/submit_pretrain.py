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
    "pretrain_dataset": ["PHM07", "PHM09", "PHM12", "CWRU", "LASPI", "HUST", "PADERBORN"],
    "window_size": 2048,
    "window_stride": 256,
    "batch_size": 256,
    "learning_rate": 8.3e-4,
    "weight_decay": 3.6e-06,
    "epochs": 50,
}

# Method-specific parameters (overrides common params if needed)
METHOD_SPECIFIC_PARAMS = {
    "mae": {
        "script": "experiments/pretrain/pretrain_mae.py",
        "mask_ratio": 0.25,
        "loss_domain": "log",
        "revin": True,
    },
    "ijepa": {
        "script": "experiments/pretrain/pretrain_ijepa.py",
        "pretrain_dataset": "CWRU",
        "window_stride": 256,
        "enc_mask_ratio": 0.5,
        "pred_mask_ratio": 0.2,
        "momentum": 0.97,
    },
    "sajepa": {
        "script": "experiments/pretrain/pretrain_sajepa.py",
        "learning_rate": 1e-4,
        "epochs": 30,
        "downsampling_factor": 2,
        "symetric_spectrum": True,
        "momentum": 0.996,
    },
    "sap": {
        "script": "experiments/pretrain/pretrain_sap.py",
        "learning_rate": 8.3e-4,
        "weight_decay": 3.6e-06,
        "hidden_dim": 512,
        "n_layers": 8,
        "heads": 8,
        "downsampling_factor": 2,
        "symetric_spectrum": True,
        "loss_domain": "linear",
        "revin": True,
    },
    "ts2vec": {
        "script": "experiments/pretrain/pretrain_ts2vec.py",
        "pretrain_dataset": "CWRU",
        "window_stride": 256,
        "batch_size": 16,
        "weight_decay": 1e-5,
        "temperature": 0.07,
        "projection_dim": 64,
        "use_cosine_similarity": True,
        "temporal_unit": 0,
    },
    "tstcc": {
        "script": "experiments/pretrain/pretrain_tstcc.py",
        "pretrain_dataset": "CWRU",
        "window_stride": 256,
        "batch_size": 64,
        "weight_decay": 1e-5,
        "epochs": 100,
        "temperature": 0.2,
        "timesteps": 4,
    },
}

# Parameters that are boolean flags (not key=value)
BOOLEAN_FLAGS = {
    "mae": ["revin"],
    "ijepa": [],
    "sajepa": ["symetric_spectrum"],
    "sap": ["symetric_spectrum", "revin", "frequency_positions"],
    "ts2vec": ["use_cosine_similarity"],
    "tstcc": [],
}

# Parameters that accept multiple datasets (list)
LIST_PARAMS = {
    "mae": ["pretrain_dataset"],
    "ijepa": [],
    "sajepa": ["pretrain_dataset"],
    "sap": ["pretrain_dataset"],
    "ts2vec": [],
    "tstcc": [],
}


def merge_configs(method, common_params_override=None):
    """Merge common params with method-specific params."""
    config = COMMON_PARAMS.copy()
    
    # Override with method-specific params
    if method in METHOD_SPECIFIC_PARAMS:
        config.update(METHOD_SPECIFIC_PARAMS[method])
    
    # Override with CLI args if provided
    if common_params_override:
        config.update(common_params_override)
    
    return config


def build_command(method, config):
    """Build the pretrain command for a given method and config."""
    script = config.pop("script")
    cmd = f"python {script}"
    
    boolean_flags = BOOLEAN_FLAGS.get(method, [])
    list_params = LIST_PARAMS.get(method, [])
    
    for key, value in config.items():
        if key in boolean_flags:
            if value:
                cmd += f" --{key}"
        elif key in list_params:
            if isinstance(value, list):
                cmd += f" --{key} {' '.join(value)}"
            else:
                cmd += f" --{key} {value}"
        else:
            cmd += f" --{key} {value}"
    
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Run pretraining experiments for different SSL methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python submit_pretrain.py mae
  python submit_pretrain.py ijepa --epochs 30 --batch_size 128
  python submit_pretrain.py sap --learning_rate 1e-3 --hidden_dim 128 --heads 4
  python submit_pretrain.py mae --pretrain_dataset CWRU LASPI --mask_ratio 0.5
        """
    )
    parser.add_argument(
        "method",
        type=str,
        choices=METHOD_SPECIFIC_PARAMS.keys(),
        help="Pretraining method to run"
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
        "--job_name",
        type=str,
        help="SLURM job name (default: method_pretrain)"
    )
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
        help="Generate SLURM script without submitting (jean-zay mode only)"
    )
    
    # Optional arguments to override common parameters
    parser.add_argument(
        "--pretrain_dataset",
        nargs="+",
        help="Override default pretrain dataset(s)"
    )
    parser.add_argument(
        "--window_size",
        type=int,
        help="Override default window size"
    )
    parser.add_argument(
        "--window_stride",
        type=int,
        help="Override default window stride"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Override default batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Override default learning rate"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        help="Override default weight decay"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override default number of epochs"
    )
    
    # Parse known args and capture remaining args for method-specific params
    args, remaining = parser.parse_known_args()
    method = args.method
    
    # Collect CLI overrides for common parameters
    common_overrides = {}
    for param in ["pretrain_dataset", "window_size", "window_stride", "batch_size", 
                  "learning_rate", "weight_decay", "epochs"]:
        value = getattr(args, param, None)
        if value is not None:
            common_overrides[param] = value
    
    # Parse remaining args as method-specific parameters
    method_overrides = {}
    i = 0
    while i < len(remaining):
        if remaining[i].startswith("--"):
            key = remaining[i][2:]  # Remove '--'
            if i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                value = remaining[i + 1]
                # Try to convert to appropriate type
                try:
                    method_overrides[key] = int(value)
                except ValueError:
                    try:
                        method_overrides[key] = float(value)
                    except ValueError:
                        if value.lower() in ["true", "false"]:
                            method_overrides[key] = value.lower() == "true"
                        else:
                            method_overrides[key] = value
                i += 2
            else:
                # Boolean flag without value
                method_overrides[key] = True
                i += 1
        else:
            i += 1
    
    # Merge configs: common + method-specific + CLI overrides
    config = merge_configs(method, common_overrides)
    config.update(method_overrides)
    
    # Build and run the command
    cmd = build_command(method, config)
    
    if args.mode == "jean-zay":
        return submit_to_jean_zay(method, cmd, args)
    else:
        return run_local(method, cmd)


def run_local(method, cmd):
    """Run training locally."""
    print(f"Starting {method.upper()} pretraining experiments...")
    print(f"Running: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Command failed: {cmd}")
        return exit_code
    
    print(f"✓ {method.upper()} pretraining completed successfully!")
    return 0


def submit_to_jean_zay(method, cmd, args):
    """Generate and optionally submit SLURM script for Jean-Zay."""
    job_name = args.job_name or f"{method}_pretrain"
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
    print(f"\nScript content:")
    print(slurm_script)
    
    if args.submit:
        print(f"\n📤 Submitting job to Jean-Zay...")
        result = subprocess.run(["sbatch", script_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Job submitted successfully!")
            print(result.stdout)
            return 0
        else:
            print(f"❌ Job submission failed!")
            print(result.stderr)
            return 1
    elif not args.script_only:
        print(f"\nTo submit the job, run:")
        print(f"  sbatch {script_name}")
        print(f"\nOr use this command to auto-submit:")
        print(f"  python submit_pretrain.py {method} --mode jean-zay --submit")
    
    return 0


if __name__ == "__main__":
    exit(main())
