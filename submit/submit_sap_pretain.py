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

batch_size = 32
learning_rate = 0.00083
weight_decay = 3.6e-06
epochs = 50

downsampling_factor = 2 # [None ,2 ,4 or 8]
symetric_spectrum = False # [True or False]

# =========================
# Pretraining experiments
# =========================
print(f"Starting SAP pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain_sap.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--downsampling_factor {downsampling_factor} "
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