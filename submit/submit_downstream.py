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

backbone_inits = ["sap"]         # ["random", "mae", "sap", "ijepa", "sajepa", "ts2vec", "tstcc"]
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI", "ConcatDataset" ]
downstream_dataset = "CWRU"         # ["CWRU", "LASPI", "CVRTEST"]
data_ratios = [1.0]                 # [0.01, 0.05, 0.1, 0.2]
split_type = "speed_stratified"     # ["independent", "speed_stratified", "speed_load_stratified", "sample_stratified"]
finetune_options = [False]    # [False, True]
head_type = "linear"                # ["linear", "non-linear"]
seeds = [0]                         # [0, 1, 2, 3, 4]  

if backbone_inits == ["mae"]:
    # backbone_checkpoint_path = "results/pretrain/MAEModel/ConcatDataset/20260915_084951"  # Path to the pretrained backbone model checkpoint
    backbone_checkpoint_path = "results/pretrain/MAEModel/ConcatDataset/20261102_153220"

elif backbone_inits == ["sap"]:
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261028_213212"      # Linear loss w/ReVIN
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261029_070753"        # Log loss w/ReVIN
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261029_102431"          # Linear loss wo/ReVIN
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261029_134809"        # Log loss wo/ReVIN

    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261030_103328" # Log loss w/ReVIN 50 epochs pretraining
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261111_171244"
    # backbone_checkpoint_path = "results/pretrain/SAPModel/ConcatDataset/20261111_184336"

    backbone_checkpoint_path = "results/pretrain/SAPModel/CWRUDataset/20261112_151741" # Log loss w/ReVIN 50 epochs pretraining

elif backbone_inits == ["sajepa"]:
    # backbone_checkpoint_path = "results/pretrain/SAJEPAModel/ConcatDataset/20261012_104552" # batch size 256 | N=4096
    # backbone_checkpoint_path = "results/pretrain/SAJEPAModel/ConcatDataset/20261030_161746"
    # backbone_checkpoint_path = "results/pretrain/SAJEPAModel/ConcatDataset/20261103_101916"
    # backbone_checkpoint_path = "results/pretrain/SAJEPAModel/ConcatDataset/20261103_133446"
    # backbone_checkpoint_path = "results/pretrain/SAJEPAModel/ConcatDataset/20261103_170856"
    backbone_checkpoint_path = "results/pretrain/SAJEPAModel/CWRUDataset/20261105_105419"


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
        f"--batch_size 64 "
        f"--window_size 2048 "
        f"--window_stride 2048 "
        f"--data_ratio {data_ratio} "
        f"--split_type {split_type} "
        f"--learning_rate 1e-4 "
        f"--weight_decay 1e-4 "
        f"--epochs {epoch} "
        f"--head_type {head_type} "
        f"--task classification "
        f"--seed {seed} "
        f"--backbone_checkpoint_path {backbone_checkpoint_path} "  # Add this line to specify the path to the pretrained backbone checkpoint
    )

    if finetune :
        downstream_cmd += " --finetune"

    print(f"Running: {downstream_cmd}")
    exit_code = os.system(downstream_cmd)

    if exit_code != 0:
        print(f"❌ Command failed: {downstream_cmd}")
