# 

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def save_recon_frame(model, batch, epoch, frames_dir:Path):
    """Sauvegarde une image cible vs prédiction pour un échantillon fixe."""
    model.eval()
    with torch.no_grad():
        out = model(batch)
    target = batch['X_raw'][0].detach().cpu().numpy()
    pred = out['prediction'][0].detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.plot(target, label='X_raw', color='black', lw=1.2)
    plt.plot(pred, label='x_pred', color='tab:red', lw=1.0, alpha=0.8)
    plt.ylim(target.min() - 0.1 * np.ptp(target), target.max() + 0.1 * np.ptp(target))
    plt.title(f'Epoch {epoch}')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(frames_dir / f"epoch_{epoch:04d}.png", dpi=100)
    plt.close()