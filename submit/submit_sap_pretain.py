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
pretrain_dataset = ["PHM07","PHM09","PHM12","CWRU","LASPI","HUST","PADERBORN"] 
window_size = 2048
window_stride = 2048

batch_size = 256
learning_rate = 8.3e-4
weight_decay = 3.6e-06
epochs = 50

# Specific hyperparameters for the Backbone
hidden_dim = 512 # [128, 256, 512]
n_layers = 8 # [4 ,8, 16]
heads = 8 # [4, 8, 12]

# Specific hyperparameters for SAP
downsampling_factor = 2 # [None ,2 ,4 or 8]
symetric_spectrum = True # [True or False]
loss_domain = "log" # ["linear" or "log"]
revin = True # [True or False]

# =========================
# Pretraining experiments
# =========================
print(f"Starting SAP pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain/pretrain_sap.py "
    f"--pretrain_dataset {' '.join(pretrain_dataset)} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--downsampling_factor {downsampling_factor} "
    f"--loss_domain {loss_domain} "
    f"--epochs {epochs} "
    f"--learning_rate {learning_rate} "
    f"--weight_decay {weight_decay} "
    f"--hidden_dim {hidden_dim} "
    f"--n_layers {n_layers} "
    f"--heads {heads} "
)

if symetric_spectrum:
    pretrain_cmd += " --symetric_spectrum"

if revin:
    pretrain_cmd += " --revin"

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")