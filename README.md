# Self-Supervised Learning for Industrial Condition Monitoring

## Overview

This repository is a comprehensive framework for investigating **self-supervised representation learning (SSL)** applied to vibration-based fault diagnosis in industrial settings, with emphasis on **data scarcity scenarios**.

**Key Research Questions:**
- How do different SSL pretext tasks affect representation quality for mechanical fault diagnosis?
- How robust are SSL representations when only limited labeled data is available?
- What is the relative benefit of linear probing versus full fine-tuning?
- How sensitive are results to pretraining dataset choice and SSL method?

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

### Dataset Setup

Download datasets and place them in `data/raw/`:

| Dataset | Source | Fault Types |
|---------|--------|-------------|
| **CWRU** | [Case Western Reserve University](https://engineering.case.edu/bearingdatacenter) | Bearing: ball, inner race, outer race |
| **LASPI** | [Gearbox Dataset](https://www.kaggle.com/) | Gear: normal, surface, inner |
| **PADERBORN** | [University of Paderborn](https://mb.uni-paderborn.de/kat/forschung/datacenter/) | Bearing faults at multiple operating conditions |
| **PHM07/09/12** | [Prognostics and Health Management Society](https://phmsociety.org/) | Turbofan, bearing, etc. |
| **CVRTEST** | N/A | Cross-validation test set |
| **HUST** | N/A | Additional bearing data |

---

## 🚀 Quick Start Guide

### 1. Pretrain a Self-Supervised Model

Example: Train a **MAE** (Masked Autoencoders) model on CWRU dataset:

```bash
python experiments/pretrain/pretrain_mae.py \
  --pretrain_dataset CWRU \
  --batch_size 256 \
  --epochs 100 \
  --mask_ratio 0.5 \
  --learning_rate 1e-3
```

Checkpoint saved to: `results/pretrain/[timestamp]/`

### 2. Run Downstream Evaluation

Evaluate pretrained model on a downstream task (e.g., fault classification on PADERBORN):

```bash
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 0.1 \
  --epochs 30 \
  --finetune
```

Results saved to: `results/downstream/[timestamp]/`

### 3. Batch Processing (Multiple Experiments)

Use the submission scripts for systematic evaluation:

```bash
python submit/submit_downstream.py
```

Edit `submit_downstream.py` to configure:
- Backbone initialization methods
- Data ratios (0.01, 0.05, 0.1, 0.2, 1.0)
- Pretraining/downstream datasets
- Number of seeds for reproducibility

---

## 📚 Architecture & Methods

### Self-Supervised Learning Methods

The framework implements multiple SSL paradigms with method-specific configurations:

| Method | Description | Status |
|--------|-------------|--------|
| **MAE** | Masked Autoencoders | ✅ Implemented |
| **TS2Vec** | Time Series Representation Learning | ✅ Implemented |
| **TSTCC** | Time Series Contrastive Learning | ✅ Implemented |
| **iJEPA** | Image Joint Embedding Predictive Architecture | ✅ Implemented |
| **SA-iJEPA** | Spectral Adapted iJEPA | ✅ Implemented |

**Design Philosophy:** Each method is implemented as a first-class research object with explicit control over:
- Input preprocessing and augmentation
- Backbone encoder architecture
- Pretext head design
- Loss definition and optimization

### Backbone Architecture

**Primary Backbone: Vision Transformer 1D (ViT-1D)**

Adapts ViT architecture to 1D vibration spectra:
- Input: Raw or frequency-domain signals (2048 samples by default)
- Tokenization: Patching strategy with learnable embeddings
- Encoder: Transformer blocks with multi-head self-attention
- Output: Learned representation tokens

Backbones are:
- Pre-trained independently via SSL
- Saved as reusable checkpoints
- Compatible with downstream evaluation
- Support learnable normalization (ReVIN)

### Downstream Tasks

**Task-Agnostic Evaluation Design:**

Supported tasks:
- **Classification**: Fault type, fault severity, operating condition
- **Regression**: Fault size, health index, degradation rate

Probing Strategies:
- **Linear Probing**: Only downstream head is trained (backbone frozen)
- **Non-Linear Probing**: Downstream head + MLP layers (backbone frozen)
- **Full Fine-Tuning**: Entire model fine-tuned on labeled data

Each downstream head:
- Implements task-specific loss
- Separate from SSL pretraining pipeline
- Clean abstraction for fair comparison
---

## 🔬 Experimental Protocols

### Data Scarcity Protocol

Systematic evaluation under limited labeled data with controlled parameters:

**Configurable Parameters:**
- SSL pretraining method (MAE, TS2Vec, TSTCC, iJEPA, SA-iJEPA)
- Pretraining dataset (CWRU, LASPI, HUST, PADERBORN, PHM07, PHM09, PHM12, CVRTEST)
- Downstream dataset (PADERBORN, LASPI, CWRU, CVRTEST, HUST, PHM07, PHM09, PHM12)
- Labeled data ratio: [0.01, 0.05, 0.1, 0.2, 1.0]
- Probing strategy (linear / non-linear)
- Fine-tuning (False=frozen backbone vs. True=fine-tuning)
- Random seeds for reproducibility

**Design Goal:** Fair and reproducible comparisons across SSL methods

### Reproducibility

Every experiment automatically logs:
- **Configuration**: Full parameter settings (JSON)
- **Checkpoints**: Model weights (`.pth` files)
- **Training Logs**: Loss curves and metrics
- **Deterministic Seeding**: Controlled randomness for reproducibility

```bash
# Run with specific seed
python experiments/downstream/downstream_sap.py \
  --seed 42 \
  --pretrain_dataset ConcatDataset \
  --downstream_dataset PADERBORN
```

---

## 📁 Project Structure

```
.
├── models/
│   ├── backbone/           # Encoder architectures (ViT-1D, etc.)
│   ├── ssl/                # Self-supervised learning methods
│   │   ├── mae.py          # Masked Autoencoders
│   │   ├── sap.py          # Spectral Aliasing Pretext
│   │   └── ...
│   └── downstream/         # Task-specific heads
│
├── dataset/
│   ├── base.py             # Base dataset class
│   ├── cwru.py, laspi.py   # Dataset implementations
│   ├── transform/          # Data preprocessing (normalization, spectrum)
│   └── dataloader.py       # DataLoader factory
│
├── training/
│   ├── pretrain.py         # Pre-training loop
│   ├── downstream.py       # Downstream evaluation loop
│   └── ...
│
├── experiments/
│   ├── pretrain/           # Pre-training scripts
│   │   └── pretrain_sap.py, pretrain_mae.py, ...
│   ├── downstream/         # Downstream task scripts
│   │   └── downstream_sap.py, downstream_mae.py, ...
│   └── optimize/           # Hyperparameter optimization
│
├── submit/                 # Batch processing scripts
│   ├── submit_downstream.py
│   └── submit_*_pretrain.py
│
├── results/                # Outputs (automatically generated)
│   ├── pretrain/           # Checkpoints and logs
│   └── downstream/         # Results per split_type
│
├── data/
│   ├── raw/                # Raw datasets
│   └── processed/          # Preprocessed data cache
│
├── utils/                  # Utilities and helpers
├── notebooks/              # Jupyter analysis notebooks
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🔧 Configuration & Customization

### Adding a Custom Dataset

1. Create a new dataset class in `dataset/`:

```python
# dataset/mydataset.py
from dataset.base import BaseDataset

class MyDataset(BaseDataset):
    def __init__(self, args, split='train'):
        super().__init__(args, split)
    
    def load_data(self):
        # Implement data loading logic
        pass
```

2. Register in `dataset/dataloader.py`:

```python
def get_dataset(name, args, split):
    if name == "MyDataset":
        return MyDataset(args, split)
```

### Adding a Custom SSL Method

1. Create SSL class in `models/ssl/`:

```python
# models/ssl/mymethod.py
import torch.nn as nn

class MySSLMethod(nn.Module):
    def __init__(self, backbone, args_ssl):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        # Implement SSL pretext task
        pass
    
    def compute_loss(self, outputs, targets):
        # Implement loss
        pass
```

2. Register in `models/ssl/registry.py`

3. Create experiment script `experiments/pretrain/pretrain_mymethod.py`

### Hyperparameter Tuning

Use `experiments/optimize/` for automated hyperparameter search:

```bash
python experiments/optimize/optimize_sap_hyperparams.py \
  --pretrain_dataset ConcatDataset \
  --n_trials 50
```

---

## 📊 Results Interpretation

### Output Structure

After running a downstream experiment:

```
results/downstream/
└── speed_stratified/
    └── ConcatDataset_to_PADERBORN_backbone_sap_head_linear_finetune_False/
        └── 20261110_153221/
            └── data_ratio_0.1_epochs_30_seed_0/
                ├── config.json                    # Full configuration
                ├── checkpoint/
                │   ├── best_model.pth            # Best validation model
                │   └── last_model.pth
                ├── log/
                │   ├── confusion_matrix.png      # Classification results
                │   ├── metrics.csv               # Performance metrics
                │   └── loss_curves.png
                └── attention_visualizations/     # Attention maps (if enabled)
                    └── attention_spectrum_class_*.png
```

### Key Metrics

**Classification:**
- Accuracy: Overall classification rate
- F1-Score (weighted): Handles class imbalance
- Confusion Matrix: Per-class performance

**Regression:**
- MSE: Mean squared error
- R²: Coefficient of determination

### Global Results Log

All experiments also log to:

```
results/downstream/metrics.csv
```

Contains: dataset combination, backbone, data ratio, metrics for each experiment

---

## 🎯 Common Use Cases

### Comparing SSL Methods on Same Dataset

```bash
for method in mae ts2vec tstcc ijepa sajepa; do
  python experiments/pretrain/pretrain_${method}.py \
    --pretrain_dataset CWRU \
    --epochs 100
done
```

### Data Scarcity Analysis (Single Method)

```bash
for ratio in 0.01 0.05 0.1 0.2 1.0; do
  python experiments/downstream/downstream_mae.py \
    --pretrain_dataset CWRU \
    --downstream_dataset PADERBORN \
    --data_ratio $ratio \
    --epochs 30
done
```

### Cross-Dataset Transfer Evaluation

```bash
# Pretrain on CWRU, evaluate on PADERBORN
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 1.0
```

### Full Fine-Tuning vs. Linear Probing

```bash
# Linear probing (frozen backbone)
python experiments/downstream/downstream_mae.py \
  --downstream_dataset PADERBORN
  # --finetune NOT set

# Full fine-tuning
python experiments/downstream/downstream_mae.py \
  --downstream_dataset PADERBORN \
  --finetune
```

---

## 🖥️ High-Performance Computing (HPC) Integration

The framework is designed for large-scale HPC execution on **Jean Zay (IDRIS)**:

**Advantages:**
- Stateless experiment scripts (clean SLURM integration)
- Automatic checkpoint management
- Parameter sweep support

**Submission Example:**

```bash
# Launch experiments via job queue
python experiments/downstream/downstream_mae.py \
  --pretrain_dataset CWRU \
  --downstream_dataset PADERBORN \
  --data_ratio 0.1 \
  --seed 42
```

---

## 🤝 Contributing

Contributions are welcome! To add features:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add feature"`
4. Push to branch: `git push origin feature/my-feature`
5. Submit a pull request

---

## 📖 Citation

If you use this code or datasets in your research, please cite:

**BibTeX:**
```bibtex
@software{Gialis2026,
  author = {Gialis, Victor},
  title = {Self-Supervised Learning for Industrial Condition Monitoring},
  year = {2026},
  url = {https://github.com/yourusername/CWRU_MAE},
  note = {GitHub repository}
}
```

**APA:**
```
Gialis, V. (2026). Self-Supervised Learning for Industrial Condition Monitoring 
under Data Scarcity. Retrieved from https://github.com/yourusername/CWRU_MAE
```

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Case Western Reserve University**: Bearing dataset (CWRU)
- **Università della Calabria**: LASPI gearbox dataset  
- **University of Paderborn**: PADERBORN bearing dataset
- **IDRIS (Jean Zay)**: High-performance computing resources
- **CWRU PhD Advisors & Lab Members**: Scientific guidance and feedback

---

## 📧 Contact & Support

**Author:** Victor Gialis  
**Position:** Industrial Maintenance & Applied Machine Learning Research

**For issues & questions:**
- Open an issue on GitHub
- Check existing discussions
- Refer to [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines

---

## 📝 Version History

- **v1.0** (Nov 2026): Initial release
  - MAE, TS2Vec, TSTCC, iJEPA, SA-iJEPA implementations
  - Multi-dataset support (CWRU, LASPI, PADERBORN, PHM series, CVRTEST, HUST)
  - Data scarcity evaluation protocol
  - Linear & non-linear probing

---

## 🔐 Code Quality & Testing

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

Project follows PEP 8 standards with:
- Black formatter for consistent code style
- Flake8 for linting
- Type hints throughout the codebase

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

---

## 📚 Further Reading

### Key Papers (Self-Supervised Learning for Time Series)

- **MAE for Time Series**: He et al., "Masked Autoencoders Are Scalable Vision Learners" (2021)
- **SAP Method**: [Spectral Aliasing Pretext specific papers]
- **TS2Vec**: Yue et al., "TS2Vec: Towards Universal Representation Learning for Time Series" (2021)
- **iJEPA**: Barreau et al., "Revisiting Feature Prediction for Learning Visual Representations" (2024)

### Industrial Condition Monitoring

- Lei et al., "A Review on Empirical Mode Decomposition in Fault Diagnosis of Rotating Machinery" (2018)
- Tran et al., "Machine Condition Monitoring and Fault Diagnosis using Deep Learning" (2018)

### Datasets

- [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter)
- [Prognostics and Health Management Society](https://phmsociety.org/)