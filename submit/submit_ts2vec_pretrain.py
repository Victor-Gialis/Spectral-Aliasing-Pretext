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

batch_size = 16
learning_rate = 3e-4
weight_decay = 1e-5
epochs = 50

# Specific hyperparameters for TS2Vec
temperature = 0.07 # Temperature parameter for contrastive loss
projection_dim = 64 # Dimension of the projection head
use_cosine_similarity = True # Use cosine similarity instead of dot product
temporal_unit = 0 # Temporal unit for instance discrimination

# =========================
# Pretraining experiments
# =========================
print(f"Starting TS2Vec Pretraining experiments...")

pretrain_cmd = (
    f"python experiments/pretrain/pretrain_ts2vec.py "
    f"--pretrain_dataset {pretrain_dataset} "
    f"--batch_size {batch_size} "
    f"--window_size {window_size} "
    f"--window_stride {window_stride} "
    f"--temperature {temperature} "
    f"--projection_dim {projection_dim} "
    f"--temporal_unit {temporal_unit} "
    f"--epochs {epochs} "
    f"--learning_rate {learning_rate} "
    f"--weight_decay {weight_decay} "
)

if use_cosine_similarity:
    pretrain_cmd += " --use_cosine_similarity"

print(f"Running: {pretrain_cmd}")
exit_code = os.system(pretrain_cmd)

if exit_code != 0:
    print(f"❌ Command failed: {pretrain_cmd}")