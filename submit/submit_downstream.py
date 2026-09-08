import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Variables for downstream experiment

backbone_inits = ["random"] # ["random", "mae", "sap", "ijepa", "ts2vec", "tstcc"]
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI"]
downstream_dataset = "CWRU"  # ["CWRU", "LASPI", "CVRTEST"]
data_ratios = [0.2] # [0.01, 0.05, 0.1, 0.2]
split_type = "speed_stratified"  # ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"]
finetune_options = [True]
head_type = "linear"  # ["linear", "nonlinear"]
seeds = [0, 1, 2] # [0, 1, 2, 3, 4]  
# epoch = 50

# =========================
# Downstream experiments
# =========================
for backbone, data_ratio, finetune, seed in product(backbone_inits, data_ratios, finetune_options, seeds):

    if not finetune and backbone == "random":
        print(f"Skipping random backbone without finetuning (not meaningful)")
        continue
    
    # Determine epochs based on data ratio
    if data_ratio == 0.01 :
        epoch = 100 # 100 epochs for 1% data
    
    elif data_ratio == 0.05 :
        epoch = 50 # 50 epochs for 5% data

    else :
        epoch = 30 # 30 epochs for 10% and 20% data

    print(f"Starting downstream evaluation with {backbone.upper()} backbone...")
    downstream_cmd = (
        f"python experiments/downstream/downstream_{backbone}.py "
        f"--pretrain_dataset {pretrain_dataset} "
        f"--downstream_dataset {downstream_dataset} "
        f"--batch_size 256 "
        f"--window_size 2048 "
        f"--window_stride 256 "
        f"--data_ratio {data_ratio} "
        f"--split_type {split_type} "
        f"--learning_rate 1e-4 "
        f"--weight_decay 1e-4 "
        f"--epochs {epoch} "
        f"--head_type {head_type} "
        f"--task classification "
        f"--seed {seed}"
    )

    if finetune :
        downstream_cmd += " --finetune"

    print(f"Running: {downstream_cmd}")
    exit_code = os.system(downstream_cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {downstream_cmd}")
