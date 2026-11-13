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
pretrain_dataset = ["PHM07","PHM09","PHM12","CWRU","LASPI","HUST","PADERBORN"]  # ["CWRU", "LASPI"]
window_size = 2048
window_stride = 2048

batch_size = 256
learning_rate = 1e-4
weight_decay = 2e-4
epochs = 30

# Specific hyperparameters for SA-JEPA
downsampling_factor = 2 # [None ,2 ,4 or 8]
symetric_spectrum = True # [True or False]
momentum = 0.996 # Momentum for the target encoder

# =========================
# Pretraining experiments
# =========================
print(f"Starting SA-JEPA Pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain/pretrain_sajepa.py "
    f"--pretrain_dataset {' '.join(pretrain_dataset)} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--downsampling_factor {downsampling_factor} "
    f"--momentum {momentum} "
    f"--epochs {epochs} "
    f"--learning_rate {learning_rate} "
    f"--weight_decay {weight_decay} "
)

if symetric_spectrum:
    pretrain_cmd += " --symetric_spectrum"

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")