import os
from itertools import product

# Set PYTHONPATH to include current directory
pythonpath = os.getenv("PYTHONPATH", "")
if pythonpath:
    pythonpath = pythonpath + ":" + os.path.abspath(".")
else:
    pythonpath = os.path.abspath(".")
os.environ["PYTHONPATH"] = pythonpath

# Variables for pretrain experiment
pretrain_dataset = "CWRU"  # ["CWRU", "LASPI"]
window_size = 2048
window_stride = 256

batch_size = 128
learning_rate = 3.0e-4
weight_decay = 6.7e-6
epochs = 50

mask_ratio = 0.80

# =========================
# Pretraining experiments
# =========================
print(f"Starting MAE pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain_mae.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--mask_ratio {mask_ratio} "
    f"--epochs {epochs} "
    f"--learning_rate {learning_rate} "
    f"--weight_decay {weight_decay} "
)

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")