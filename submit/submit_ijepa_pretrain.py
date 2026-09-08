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

batch_size = 256
learning_rate = 3e-4
weight_decay = 2e-4
epochs = 50

# Specific hyperparameters for IJEPA
enc_mask_ratio = 0.5 # Masking ratio for the context encoder
pred_mask_ratio = 0.2 # Masking ratio for the prediction encoder
momentum = 0.97 # Momentum for the target encoder

# =========================
# Pretraining experiments
# =========================
print(f"Starting IJEPA Pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain_ijepa.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--enc_mask_ratio {enc_mask_ratio} "
    f"--pred_mask_ratio {pred_mask_ratio} "
    f"--momentum {momentum} "
    f"--epochs {epochs} "
    f"--learning_rate {learning_rate} "
    f"--weight_decay {weight_decay} "
)

# if symetric_spectrum:
#     pretrain_cmd += " --symetric_spectrum"

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")