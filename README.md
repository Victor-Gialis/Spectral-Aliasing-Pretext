# Self-Supervised Learning for Industrial Condition Monitoring

## Overview

This repository is a comprehensive framework for investigating **self-supervised representation learning (SSL)** applied to vibration-based fault diagnosis in industrial settings, with emphasis on **data scarcity scenarios**.

**Key Research Questions:**
- How do different SSL pretext tasks affect representation quality for mechanical fault diagnosis?
- How robust are SSL representations when only limited labeled data is available?
- What is the relative benefit of linear probing versus full fine-tuning?
- How sensitive are results to pretraining dataset choice and SSL method?

**Supported SSL Methods:**
- **MAE** (Masked Autoencoders)
- **SAP** (Spectral Aliasing Pretext)
- **TS2Vec** (Time Series Representation Learning)
- **TSTCC** (Time Series Contrastive Clustering)
- **iJEPA** (Image Joint Embedding Predictive Architecture)
- **SA-iJEPA** (Spectral Adapted iJEPA)

**Supported Datasets:**
- CWRU (Case Western Reserve University - Bearing data)
- LASPI (Gearbox data)
- PADERBORN (Bearing faults at multiple operating conditions)
- PHM07, PHM09, PHM12 (Prognostics and Health Management datasets)
- CVRTEST (Cross-validation test set)
- HUST (Additional bearing data)

This project serves as the experimental framework for doctoral research in industrial maintenance and applied machine learning.

---

## 📋 Installation & Setup

### Prerequisites
- **Python 3.10+**
- **CUDA 11.8+** (for GPU support; CPU-only is supported)
- **~15 GB disk space** (for all datasets and pretrained models)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/CWRU_MAE.git
cd CWRU_MAE

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install as editable package for development
pip install -e .
```

**Main Dependencies:**
- PyTorch 2.0+ with CUDA 11.8+ support
- TorchVision, TorchAudio
- scikit-learn, pandas, numpy
- PyYAML, scipy
- Weights & Biases (wandb) for experiment tracking
- Tensorboard for visualization

### Dataset Setup

Datasets are stored in `data/raw/` directory. Currently supported datasets:

| Dataset | Type | Fault Types | Status |
|---------|------|-------------|--------|
| **CWRU** | Bearing | Ball, inner race, outer race defects | ✅ Implemented |
| **LASPI** | Gearbox | Normal, surface, inner race defects | ✅ Implemented |
| **PADERBORN** | Bearing | Faults at multiple operating conditions | ✅ Implemented |
| **PHM07** | Turbofan | Engine degradation | ✅ Implemented |
| **PHM09** | Bearing | Run-to-failure degradation | ✅ Implemented |
| **PHM12** | Equipment | Health monitoring data | ✅ Implemented |
| **CVRTEST** | Cross-validation | Test set for validation | ✅ Implemented |
| **HUST** | Bearing | Additional bearing fault data | ✅ Implemented |

**Data Organization:**
```
data/raw/
├── CWRU/
├── LASPI/
├── PADERBORN/
├── PHM07/
├── PHM09/
├── PHM12/
├── CVRTEST/
└── HUST/

data/processed/
└── [Cached preprocessed data]
```

---

## 🚀 Quick Start Guide

### 1. Pretrain a Self-Supervised Model

Pretrain a model on a specific dataset. Currently available methods: MAE, SAP, TS2Vec, TSTCC, iJEPA, SA-iJEPA

**Example: Train MAE on CWRU dataset:**

```bash
python experiments/pretrain/pretrain_mae.py \
  --pretrain_dataset CWRU \
  --batch_size 256 \
  --epochs 50 \
  --mask_ratio 0.25 \
  --learning_rate 3.0e-4 \
  --window_size 2048
```

**Available pretrain scripts:**
- `experiments/pretrain/pretrain_mae.py`
- `experiments/pretrain/pretrain_sap.py`
- `experiments/pretrain/pretrain_ts2vec.py`
- `experiments/pretrain/pretrain_tstcc.py`
- `experiments/pretrain/pretrain_ijepa.py`
- `experiments/pretrain/pretrain_sajepa.py`

Checkpoint saved to: `results/pretrain/[timestamp]/checkpoint/`

### 2. Run Downstream Evaluation

Evaluate a pretrained (or randomly initialized) model on downstream classification tasks:

```bash
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 0.1 \
  --epochs 30 \
  --finetune
```

**Available downstream scripts:**
- `experiments/downstream/downstream_mae.py`
- `experiments/downstream/downstream_sap.py`
- `experiments/downstream/downstream_ts2vec.py`
- `experiments/downstream/downstream_tstcc.py`
- `experiments/downstream/downstream_ijepa.py`
- `experiments/downstream/downstream_sajepa.py`
- `experiments/downstream/downstream_random.py` (Random initialization baseline)

Results saved to: `results/downstream/[split_type]/[experiment_config]/[timestamp]/`

### 3. Batch Processing & Hyperparameter Optimization

**Run batch pretraining:**
```bash
python submit/submit_pretrain.py <method>
# Example: python submit/submit_pretrain.py mae
```

**Run batch downstream evaluation:**
```bash
python submit/submit_downstream.py
```

Edit `submit/submit_pretrain.py` and `submit/submit_downstream.py` to configure:
- Pretraining datasets
- Data ratios (0.01, 0.05, 0.1, 0.2, 1.0)
- SSL methods and hyperparameters
- Number of random seeds for reproducibility

**Hyperparameter optimization (via Optuna):**
```bash
python experiments/optimize/optimize_mae_hyperparams.py \
  --pretrain_dataset CWRU \
  --n_trials 50
```

---

## 📚 Architecture & Methods

### Self-Supervised Learning Methods

All SSL methods are implemented in `models/ssl/` with consistent interface:

| Method | File | Description | Status |
|--------|------|-------------|--------|
| **MAE** | `mae.py` | Masked Autoencoders for time series | ✅ Implemented |
| **SAP** | `sap.py` | Spectral Aliasing Pretext task | ✅ Implemented |
| **TS2Vec** | `ts2vec.py` | Time Series Representation Learning | ✅ Implemented |
| **TSTCC** | `tstcc.py` | Time Series Contrastive Clustering | ✅ Implemented |
| **iJEPA** | `ijepa.py` | Joint Embedding Predictive Architecture | ✅ Implemented |
| **SA-iJEPA** | `sajepa.py` | Spectral Adapted iJEPA | ✅ Implemented |

### Backbone Architectures

**Available backbones** in `models/backbone/`:

| Backbone | File | Input | Description |
|----------|------|-------|-------------|
| **ViT-1D** | `vit1d.py` | Time or frequency domain | Vision Transformer adapted to 1D signals (default) |
| **CNN** | `cnn.py` | Time or frequency domain | Standard convolutional neural network |
| **Dilated CNN** | `cnn.py` | Time or frequency domain | CNN with dilated convolutions |

**Features:**
- Configurable domain (time/frequency)
- Optional learnable normalization (ReVIN)
- Scaling method options: log, power, standard
- Patch embedding strategy for ViT

**Default configuration:**
```python
args_backbone = SimpleNamespace(
    model="vit1d",        # Backbone type
    domain="time",        # Input domain
    scaling_method=None   # Scaling method
)
```

### Downstream Tasks

**Classification:**
- Multi-class fault classification (ball, inner race, outer race, etc.)
- Operating condition classification (speed, load)

**Regression:**
- Health index prediction
- Remaining useful life (RUL) estimation
- Fault severity estimation

**Probing strategies** in `models/downstream/`:
- **Linear Probing** (`classification.py`, `regression.py`): Only downstream head trained
- **Full Fine-tuning**: Entire model updated with low learning rate
- **Non-linear Probing**: MLP layers for more expressive downstream head
---

## 🔬 Experimental Protocols

### Data Scarcity Protocol

Systematic evaluation under limited labeled data:

**Key Configuration Parameters:**
```python
{
  "pretrain_dataset": "CWRU | LASPI | PADERBORN | PHM07 | PHM09 | PHM12 | CVRTEST | HUST | ConcatDataset",
  "downstream_dataset": "CWRU | LASPI | PADERBORN | CVRTEST",
  "ssl_method": "mae | sap | ts2vec | tstcc | ijepa | sajepa | random",
  "data_ratio": [0.01, 0.05, 0.1, 0.2, 1.0],
  "split_type": "speed_stratified | speed_load_stratified",
  "finetune": false,  # Linear probing vs fine-tuning
  "seed": 0  # Reproducibility
}
```

**Default hyperparameters by method:**
- **MAE**: mask_ratio=0.25, loss_domain="log"
- **SAP**: downsampling_factor=2, symmetric_spectrum=True
- **TS2Vec**: tau=0.5, hidden_dims=64
- **TSTCC**: hidden_dims=64, epsilon=0.5
- **iJEPA**: enc_mask_ratio=0.5, pred_mask_ratio=0.2, momentum=0.97
- **SA-iJEPA**: momentum=0.996, epochs=30

### Data Splits

**Split types supported in `dataset/split_data_factory.py`:**
1. **speed_stratified**: Stratified by operating speed
2. **speed_load_stratified** (LASPI only): Stratified by speed AND load conditions

**Train/Valid/Test ratio:** 60% / 20% / 20% (configurable)

### Reproducibility

Every experiment logs:
- **config.json**: Complete hyperparameters
- **checkpoint/**: Model weights (.pth)
- **log/**: Metrics CSV, confusion matrices, loss curves
- **Deterministic seeding**: torch.manual_seed, np.random.seed, random.seed

```bash
# Run with specific seed for reproducibility
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --seed 42
```

---

## 📁 Project Structure

```
.
├── models/                 # Model implementations
│   ├── __init__.py
│   ├── backbone/           # Encoder architectures
│   │   ├── vit1d.py        # Vision Transformer 1D (default)
│   │   ├── cnn.py          # CNN and Dilated CNN variants
│   │   ├── registry.py     # Backbone factory
│   │   └── __pycache__/
│   │
│   ├── ssl/                # Self-supervised learning methods
│   │   ├── base.py         # Base SSL class
│   │   ├── mae.py          # Masked Autoencoders
│   │   ├── sap.py          # Spectral Aliasing Pretext
│   │   ├── ts2vec.py       # TS2Vec method
│   │   ├── tstcc.py        # TSTCC contrastive method
│   │   ├── ijepa.py        # iJEPA method
│   │   ├── sajepa.py       # Spectral Adapted iJEPA
│   │   ├── registry.py     # SSL method factory
│   │   └── __pycache__/
│   │
│   └── downstream/         # Task-specific heads
│       ├── base.py         # Base downstream class
│       ├── classification.py # Classification heads
│       ├── regression.py    # Regression heads
│       ├── inference_downstream.py # Inference utilities
│       ├── registry.py     # Downstream factory
│       └── __pycache__/
│
├── dataset/                # Dataset implementations
│   ├── __init__.py
│   ├── base.py            # BaseDataset class
│   ├── cwru.py            # CWRU bearing dataset
│   ├── laspi.py           # LASPI gearbox dataset
│   ├── paderborn.py       # PADERBORN bearing dataset
│   ├── phm07.py           # PHM07 turbofan dataset
│   ├── phm09.py           # PHM09 bearing dataset
│   ├── phm12.py           # PHM12 equipment dataset
│   ├── cvrtest.py         # CVRTEST validation set
│   ├── hust.py            # HUST bearing dataset
│   ├── dataloader.py      # DataLoader factory
│   ├── split_data_factory.py # Train/valid/test splitting
│   │
│   └── transform/          # Data preprocessing
│       ├── __init__.py
│       ├── normalization.py # ReVIN normalization
│       ├── spectrum.py      # FFT/spectral transforms
│       ├── undersampling.py # Class balancing
│       └── __pycache__/
│
├── training/              # Training loops
│   ├── __init__.py
│   ├── pretrain.py        # Pre-training function
│   ├── downstream.py      # Downstream training function
│   └── __pycache__/
│
├── experiments/           # Experiment scripts
│   ├── __init__.py
│   │
│   ├── pretrain/          # Pre-training experiments
│   │   ├── pretrain_mae.py
│   │   ├── pretrain_sap.py
│   │   ├── pretrain_ts2vec.py
│   │   ├── pretrain_tstcc.py
│   │   ├── pretrain_ijepa.py
│   │   └── pretrain_sajepa.py
│   │
│   ├── downstream/        # Downstream evaluation experiments
│   │   ├── downstream_mae.py
│   │   ├── downstream_sap.py
│   │   ├── downstream_ts2vec.py
│   │   ├── downstream_tstcc.py
│   │   ├── downstream_ijepa.py
│   │   ├── downstream_sajepa.py
│   │   └── downstream_random.py
│   │
│   └── optimize/          # Hyperparameter optimization
│       ├── optimize_mae_hyperparams.py
│       ├── optimize_sap_hyperparams.py
│       └── optimize_ts2vec_hyperparams.py
│
├── submit/                # Batch processing scripts
│   ├── submit_pretrain.py          # Batch pretraining jobs
│   └── submit_downstream.py        # Batch downstream jobs
│
├── results/               # Experiment outputs (auto-generated)
│   ├── pretrain/          # Pre-training checkpoints
│   ├── downstream/        # Downstream results
│   ├── linear_probing/    # Linear probing specific
│   ├── mse_vs_f1_analysis/# Analysis outputs
│   └── optimization/      # Hyperparameter optimization results
│
├── data/                  # Datasets
│   ├── raw/               # Raw data storage
│   │   ├── CWRU/
│   │   ├── LASPI/
│   │   ├── PADERBORN/
│   │   ├── PHM07/
│   │   ├── PHM09/
│   │   ├── PHM12/
│   │   ├── CVRTEST/
│   │   └── HUST/
│   └── processed/         # Cached preprocessed data
│
├── docs/                  # Documentation & results
│   ├── index.html         # Main results page
│   ├── cwru_results.html  # CWRU experiment results
│   ├── laspi_results.html # LASPI experiment results
│   ├── cwru/              # Per-class analysis (CWRU)
│   └── laspi/             # Per-class analysis (LASPI)
│
├── notebooks/             # Jupyter notebooks for analysis
│   ├── ablation.ipynb     # Ablation studies
│   ├── plot_results.ipynb # Results visualization
│   └── test.ipynb         # Testing/debugging
│
├── examples/              # Example scripts
│   ├── example_mse_f1_analysis.py
│   └── visualize_paderborn_spectra.py
│
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── analysis_metrics.py     # Metric calculations
│   ├── quick_mse_f1_analysis.py # Quick analysis
│   ├── recon_gif.py            # Reconstruction visualization
│   ├── transformer_blocks.py    # Transformer utilities
│   └── __pycache__/
│
├── config/                # Configuration files
│   ├── paths.py           # Path configuration
│   ├── README.md
│   └── __pycache__/
│
├── __pycache__/
├── requirements.txt       # Python dependencies
├── setup.py              # Package setup
├── inference.ipynb       # Inference notebook
└── README.md             # This file
```

---

## 🔧 Configuration & Customization

### Common Experiment Configurations

#### MAE Configuration
```python
args_ssl = SimpleNamespace(
    method="mae",
    mask_ratio=0.25,      # 25% of patches masked
    loss_domain="log",    # Log domain loss for frequency analysis
    revin=True,           # Reversible instance normalization
)
```

#### SAP (Spectral Aliasing Pretext) Configuration
```python
args_ssl = SimpleNamespace(
    method="sap",
    downsampling_factor=2,
    symmetric_spectrum=True,
)
```

#### TS2Vec Configuration
```python
args_ssl = SimpleNamespace(
    method="ts2vec",
    tau=0.5,
    hidden_dims=64,
)
```

#### iJEPA Configuration
```python
args_ssl = SimpleNamespace(
    method="ijepa",
    enc_mask_ratio=0.5,   # Encoder input masking
    pred_mask_ratio=0.2,  # Predictor masking
    momentum=0.97,        # EMA momentum
)
```

### Adding a Custom Dataset

1. Create dataset class in `dataset/`:

```python
# dataset/mydataset.py
from dataset.base import BaseDataset

class MydatasetDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().__init__(args, split)
    
    def load_data(self):
        # Implement data loading logic
        self.data = ...
        self.labels = ...
        pass
```

2. Register in `dataset/dataloader.py` (auto-discovered via naming convention)

3. Use in experiments:
```bash
python experiments/downstream/downstream_mae.py \
  --downstream_dataset Mydataset
```

### Adding a Custom SSL Method

1. Create SSL class in `models/ssl/`:

```python
# models/ssl/mymethod.py
from models.ssl.base import BaseSSLMethod
import torch.nn as nn

class MyMethodModel(BaseSSLMethod):
    def __init__(self, backbone, args_ssl):
        super().__init__(backbone, args_ssl)
        # Initialize pretext head
        self.head = nn.Linear(...)
    
    def forward(self, x):
        # Implement SSL pretext task
        embeddings = self.backbone(x)
        outputs = self.head(embeddings)
        return outputs
    
    def compute_loss(self, outputs, targets):
        # Implement custom loss
        loss = ...
        return loss
```

2. Register in `models/ssl/registry.py`

3. Create experiment script `experiments/pretrain/pretrain_mymethod.py` (follows existing patterns)

### Domain Configuration

Models support both **time** and **frequency** domain input:

```python
args_backbone = SimpleNamespace(
    model="vit1d",
    domain="time",        # or "frequency"
    scaling_method="log"  # or "power", "standard", None
)
```

**Scaling methods:**
- **log**: Log-scale for emphasizing small values
- **power**: Power-law scaling
- **standard**: Standard normalization (zero-mean, unit-variance)
- **None**: No scaling

### Output Structure

After running downstream experiments:

```
results/downstream/
└── <split_type>/
    └── <pretrain>_to_<downstream>_backbone_<method>_head_<head_type>_finetune_<bool>/
        └── <timestamp>/
            └── data_ratio_<ratio>_epochs_<n>_seed_<seed>/
                ├── config.json              # Complete configuration
                ├── checkpoint/
                │   ├── best_model.pth      # Best validation checkpoint
                │   └── last_model.pth
                ├── log/
                │   ├── metrics.csv         # Train/val metrics
                │   ├── confusion_matrix.png
                │   └── loss_curves.png
                └── attention_visualizations/ (optional)
                    └── attention_*.png
```

### Key Output Metrics

**Classification:**
- Accuracy (overall)
- Precision, Recall, F1-score (per-class)
- Confusion matrix
- ROC-AUC (if binary)

**Regression:**
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

---

## 📊 Results & Analysis

### Result Organization

All experimental results are organized by split type and configuration:

```
results/downstream/
├── speed_stratified/              # Default split
│   ├── CWRU_to_PADERBORN_...
│   ├── LASPI_to_CWRU_...
│   └── ...
└── speed_load_stratified/         # For LASPI data
    ├── LASPI_to_PADERBORN_...
    └── ...
```

### Analyzing Results

**Global metrics CSV:**
```bash
# All downstream results consolidated
cat results/downstream/metrics.csv
```

**Per-experiment metrics:**
```bash
# Individual experiment results
cat results/downstream/speed_stratified/<config>/<timestamp>/data_ratio_*/log/metrics.csv
```

### Key Performance Metrics

| Metric | Usage | Notes |
|--------|-------|-------|
| **Accuracy** | Classification | Overall correct predictions |
| **F1-Score (Weighted)** | Classification | Handles class imbalance |
| **Precision/Recall** | Classification | Per-class performance |
| **Confusion Matrix** | Classification | Error patterns visualization |
| **MSE / RMSE** | Regression | Prediction error magnitude |
| **R² Score** | Regression | Variance explained |
| **MAE** | Regression | Mean absolute error |

### Visualization Tools

**Available visualization outputs:**
- **Loss curves**: Training and validation loss over epochs
- **Confusion matrices**: PNG heatmaps for classification results
- **Attention maps**: Transformer attention visualizations (if enabled)
- **Spectrograms**: FFT analysis plots
- **Reconstruction GIFs**: For analyzing model reconstructions

### Interactive Results

**Jupyter notebooks for analysis:**
- `notebooks/plot_results.ipynb`: General results visualization
- `notebooks/ablation.ipynb`: Ablation study analysis
- `notebooks/test.ipynb`: Quick testing and debugging

**Generate documentation:**
```bash
# Results are saved to docs/ for web viewing
# Open docs/index.html in browser for interactive results
```

---

## 🎯 Common Use Cases & Examples

### 1. Compare Multiple SSL Methods on Same Dataset

```bash
# Pretrain multiple methods on CWRU
for method in mae sap ts2vec tstcc ijepa sajepa; do
  python experiments/pretrain/pretrain_${method}.py \
    --pretrain_dataset CWRU \
    --batch_size 256 \
    --epochs 50
done
```

### 2. Data Scarcity Analysis

Evaluate model performance with varying amounts of labeled data:

```bash
# Test MAE on PADERBORN with different data ratios
for ratio in 0.01 0.05 0.1 0.2 1.0; do
  python experiments/downstream/downstream_mae.py \
    --pretrain_dataset CWRU \
    --downstream_dataset PADERBORN \
    --data_ratio $ratio \
    --epochs 30 \
    --seed 42
done
```

### 3. Cross-Dataset Transfer Evaluation

Evaluate generalization across different equipment types:

```bash
# Pretrain on CWRU bearing data, test on PADERBORN
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 1.0 \
  --epochs 50
```

### 4. Linear Probing vs Fine-Tuning

Compare frozen vs trainable backbone:

```bash
# Linear probing (backbone frozen)
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --finetune False

# Full fine-tuning
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --finetune True
```

### 5. Baseline Comparison (Random Initialization)

Compare SSL methods against random initialization:

```bash
python experiments/downstream/downstream_random.py \
  --downstream_dataset PADERBORN \
  --data_ratio 0.1 \
  --epochs 30
```

### 6. Hyperparameter Optimization

Automatically search optimal hyperparameters using Optuna:

```bash
python experiments/optimize/optimize_mae_hyperparams.py \
  --pretrain_dataset CWRU \
  --n_trials 100
```

### 7. Analysis & Visualization

**Quick MSE vs F1 analysis:**
```bash
python utils/quick_mse_f1_analysis.py --dataset PADERBORN
```

**Visualize spectrograms:**
```bash
python examples/visualize_paderborn_spectra.py
```

**Plot results from notebooks:**
```bash
jupyter notebook notebooks/plot_results.ipynb
```

### 8. Using Inference Notebook

For quick inference and visualization:

```bash
jupyter notebook inference.ipynb
```

Configure in first cell:
```python
DOWNSTREAM_DATASET = "PADERBORN"
BACKBONE_INIT = "sap"  # or "mae", "random"
FINETUNE_OPTION = False
```

---

## 🖥️ Advanced Features

### Experiment Tracking

**Weights & Biases Integration:**
```bash
# Automatically tracks all experiments to wandb
# Configure in experiment scripts by setting WANDB_PROJECT
export WANDB_PROJECT="industrial-condition-monitoring"
python experiments/downstream/downstream_mae.py ...
```

**Tensorboard Support:**
```bash
# Monitor training in real-time
tensorboard --logdir results/downstream/
```

### High-Performance Computing (HPC)

The framework is optimized for large-scale HPC execution:

**Features:**
- Stateless experiment scripts (clean SLURM integration)
- Automatic checkpoint management and recovery
- Distributed training support via PyTorch DDP
- Parameter sweep via `submit/submit_*.py` scripts

**Example SLURM submission:**
```bash
#!/bin/bash
#SBATCH --job-name=downstream_mae
#SBATCH --gpus=1
#SBATCH --time=4:00:00

python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 0.1 \
  --seed $SLURM_ARRAY_TASK_ID
```

### Distributed Training

For multi-GPU training (future support):
```python
# Models support DistributedDataParallel
import torch.nn as nn
model = nn.parallel.DistributedDataParallel(model, device_ids=[0])
```

### Memory & Compute Optimization

**Default resource requirements:**
- **GPU Memory**: ~8GB for ViT-1D (batch_size=256)
- **CPU**: 8-16 cores recommended
- **Training time**: ~1-2 hours per pretraining run (on NVIDIA A100)

---

## 🤝 Contributing

Contributions are welcome! To add features or improve the framework:

1. **Fork & Clone:**
   ```bash
   git clone https://github.com/yourusername/CWRU_MAE.git
   cd CWRU_MAE
   ```

2. **Create Feature Branch:**
   ```bash
   git checkout -b feature/my-feature
   ```

3. **Install Development Dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Code Style:**
   - Follow PEP 8 standards
   - Use Black for formatting: `black models/ dataset/ training/`
   - Use isort for imports: `isort models/ dataset/ training/`
   - Run Flake8: `flake8 models/ dataset/ training/`

5. **Testing:**
   ```bash
   pytest tests/ -v
   ```

6. **Commit & Push:**
   ```bash
   git add .
   git commit -m "Add feature: description"
   git push origin feature/my-feature
   ```

7. **Submit Pull Request**

### Testing New Methods

When implementing new SSL methods:

1. Add method to `models/ssl/`
2. Register in `models/ssl/registry.py`
3. Create experiment script in `experiments/pretrain/`
4. Test on small dataset first:
   ```bash
   python experiments/pretrain/pretrain_mymethod.py \
     --pretrain_dataset CWRU \
     --batch_size 32 \
     --epochs 5  # Short test run
   ```

### Reporting Issues

Please include:
- Python version and OS
- PyTorch and CUDA versions
- Minimal reproducible example
- Stack trace and error message

---

## 📖 Citation

If you use this code, datasets, or methodology in your research, please cite:

**BibTeX:**
```bibtex
@software{CWRU_MAE_2026,
  author = {Gialis, Victor},
  title = {Self-Supervised Learning for Industrial Condition Monitoring under Data Scarcity},
  year = {2026},
  url = {https://github.com/yourusername/CWRU_MAE},
  note = {GitHub Repository}
}
```

**APA:**
```
Gialis, V. (2026). Self-supervised learning for industrial condition monitoring 
under data scarcity. Retrieved from https://github.com/yourusername/CWRU_MAE
```

### Dataset Citations

If using specific datasets, please also cite:

**CWRU Bearing Data:**
```bibtex
@misc{CWRU_bearing,
  title = {Bearing Data Center},
  author = {Case Western Reserve University},
  url = {https://engineering.case.edu/bearingdatacenter},
  year = {2024}
}
```

**PADERBORN Bearing Data:**
```bibtex
@article{PADERBORN,
  title = {A public bearing vibration dataset},
  author = {University of Paderborn},
  url = {https://mb.uni-paderborn.de/kat/forschung/datacenter/},
  year = {2024}
}
```

**PHM Datasets:**
```bibtex
@misc{PHM_datasets,
  title = {PHM Data Repository},
  author = {PHM Society},
  url = {https://phmsociety.org/},
  year = {2024}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

The datasets used in this research retain their original licenses:
- **CWRU**: Public dataset
- **LASPI/PADERBORN/PHM**: Check respective sources for license information

---

## 🙏 Acknowledgments

- **Case Western Reserve University**: Bearing dataset and infrastructure
- **Università della Calabria**: LASPI gearbox dataset  
- **University of Paderborn**: PADERBORN bearing dataset
- **PHM Society**: Public datasets and community support
- **PyTorch Team**: Deep learning framework
- **Research Community**: Feedback and improvements

---

## 📧 Contact & Support

**Author:** Victor Gialis  
**Research Focus:** Industrial Maintenance & Applied Machine Learning

**For issues and questions:**
- 📋 Open an issue on GitHub
- 💬 Check existing discussions and Q&A
- 📖 Refer to this README and documentation
- 🐛 Include minimal reproducible examples when reporting bugs

**For research inquiries:**
- Cite this work if using in your research
- Provide feedback on methodology and results
- Collaborate on extensions and improvements

---

## 📝 Release Notes

### Version 0.1.0 (November 2026)

**Initial Release Features:**
- ✅ 6 SSL methods (MAE, SAP, TS2Vec, TSTCC, iJEPA, SA-iJEPA)
- ✅ 3 backbone architectures (ViT-1D, CNN, Dilated CNN)
- ✅ 8 industrial bearing/equipment datasets
- ✅ Data scarcity evaluation protocol
- ✅ Linear probing and fine-tuning support
- ✅ Hyperparameter optimization framework
- ✅ Comprehensive experiment tracking
- ✅ Interactive visualization tools

**Planned Enhancements:**
- Multi-GPU distributed training support
- Additional SSL methods (SimCLR, BYOL, SwAV)
- Domain adaptation techniques
- Real-time inference pipeline
- Interactive web dashboard

---

## 🔒 Code Quality & Standards

### Quality Assurance

- **Type Hints**: All functions have type annotations
- **Docstrings**: Comprehensive documentation for all classes/functions
- **Code Style**: PEP 8 compliant with Black formatter
- **Testing**: Unit tests and integration tests (pytest)
- **Linting**: Flake8 and isort for code quality

### Pre-commit Setup

```bash
pip install pre-commit
pre-commit install

# Manual run
pre-commit run --all-files
```

**Configured checks:**
- Black (formatting)
- Flake8 (linting)
- isort (import sorting)
- Type checking (via Pylance)

---

## 📚 Additional Resources

### Key Reference Papers

**Self-Supervised Learning Methods:**
- He et al. "Masked Autoencoders Are Scalable Vision Learners" (2021)
- Yue et al. "TS2Vec: Towards Universal Representation Learning for Time Series" (2021)
- Barreau et al. "Revisiting Feature Prediction for Learning Visual Representations" (2024)
- Chen et al. "Contrastive Learning of Structured World Models" (2020)

**Industrial Condition Monitoring:**
- Lei et al. "Applications of intelligent machinery condition monitoring" (2020)
- Tran et al. "Machine Condition Monitoring and Fault Diagnosis using Deep Learning" (2018)
- Zhao et al. "Deep Learning Algorithms for Rotating Machinery Anomaly Detection" (2016)

**Time Series Analysis:**
- Yeh et al. "Time Series Anomaly Detection with Autoencoders" (2019)
- Zerveas et al. "A Transformer-based Framework for Multivariate Time Series Forecasting" (2021)

### External Datasets

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [University of Paderborn DataCenter](https://mb.uni-paderborn.de/kat/forschung/datacenter/)
- [PHM Society Data Repository](https://phmsociety.org/)
- [Kaggle - Rotating Machinery Datasets](https://www.kaggle.com/search?q=bearing%20fault)

### Community & Forums

- [PyTorch Discuss Forum](https://discuss.pytorch.org/)
- [Time Series Deep Learning](https://github.com/topics/time-series-analysis)
- [Condition Monitoring Research](https://github.com/topics/condition-monitoring)

---

## ⚡ Quick Reference

### Common Commands

```bash
# Pretraining
python experiments/pretrain/pretrain_mae.py --pretrain_dataset CWRU --epochs 50

# Downstream evaluation  
python experiments/downstream/downstream_mae.py --pretrain_dataset CWRU --downstream_dataset PADERBORN --data_ratio 0.1

# Batch jobs
python submit/submit_pretrain.py mae
python submit/submit_downstream.py

# Hyperparameter search
python experiments/optimize/optimize_mae_hyperparams.py --pretrain_dataset CWRU --n_trials 100

# Analysis
python utils/quick_mse_f1_analysis.py --dataset PADERBORN
jupyter notebook inference.ipynb
```

### Key Configuration Parameters

| Parameter | Default | Options | Purpose |
|-----------|---------|---------|---------|
| `--model` | `vit1d` | vit1d, cnn, dilated_cnn | Backbone architecture |
| `--domain` | `time` | time, frequency | Input domain |
| `--batch_size` | 256 | 32-512 | Training batch size |
| `--epochs` | 50 | 1-500 | Number of training epochs |
| `--learning_rate` | 3e-4 | 1e-5 to 1e-2 | Optimizer learning rate |
| `--data_ratio` | 1.0 | 0.01-1.0 | Fraction of labeled data |
| `--finetune` | False | True/False | Fine-tune backbone |
| `--window_size` | 2048 | 1024-4096 | Sample window length |
| `--mask_ratio` | 0.25 | 0.1-0.75 | MAE masking ratio |
| `--seed` | 0 | 0-999 | Random seed |

### Python Version & Dependencies

```bash
python --version  # Requires Python 3.10+

# Check installation
python -c "import torch; print(torch.__version__)"
python -c "import torchvision; print(torchvision.__version__)"

# Verify GPU support
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 🚀 Getting Started (5-Minute Quick Start)

```bash
# 1. Clone and setup
git clone https://github.com/yourusername/CWRU_MAE.git
cd CWRU_MAE
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pretrain a model (5 epochs for testing)
python experiments/pretrain/pretrain_mae.py \
  --pretrain_dataset CWRU \
  --batch_size 64 \
  --epochs 5 \
  --learning_rate 1e-3

# 3. Evaluate on downstream task
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 1.0 \
  --epochs 10

# 4. Check results
cat results/downstream/speed_stratified/*/*/log/metrics.csv

# 5. Run inference
jupyter notebook inference.ipynb
```

---

## 🎓 Learning Path

**Beginner:**
1. Read this README and understand the project scope
2. Run the 5-minute quick start
3. Explore `inference.ipynb` for hands-on examples
4. Check `notebooks/plot_results.ipynb` for visualization

**Intermediate:**
1. Implement data scarcity experiments (Section 🎯)
2. Compare multiple SSL methods
3. Analyze results using provided notebooks
4. Run hyperparameter optimization

**Advanced:**
1. Add custom SSL methods (Section 🔧)
2. Integrate new datasets
3. Modify backbone architectures
4. Contribute to the framework

---

## ❓ Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```bash
# Reduce batch size
python experiments/downstream/downstream_mae.py --batch_size 32 ...
# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**Issue: Dataset not found**
```bash
# Check data directory structure
ls data/raw/CWRU
# Ensure raw data is downloaded and in correct location
```

**Issue: Import errors**
```bash
# Reinstall with development dependencies
pip install -e ".[dev]"

# Check Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -c "import models; print('OK')"
```

**Issue: Checkpoint loading failed**
```bash
# Clear cache and retry
rm -rf __pycache__ data/processed/*
python experiments/downstream/downstream_mae.py ...
```

### Getting Help

1. **Search closed issues**: Check if your problem is already solved
2. **Check error message**: Usually indicates root cause and solution
3. **Review configuration**: Verify all parameters are correct
4. **Enable verbose logging**: Add `--verbose` flag for detailed output
5. **Post GitHub issue**: Include full error trace and minimal example

---

## 🔗 Related Projects

- [TimeSeriesAnomalyDetection](https://github.com/topics/anomaly-detection)
- [RotatingMachineryFaultDiagnosis](https://github.com/topics/fault-diagnosis)
- [SelfSupervisedLearning](https://github.com/topics/self-supervised-learning)
- [IndustrialAI](https://github.com/topics/industrial-ai)

---

**Last Updated:** November 2026  
**Maintainer:** Victor Gialis  
**Status:** Active Development ✅