import torch
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.amp import GradScaler, autocast
import mlflow
import tqdm

def set_seed(seed: int = 42) -> None:
    """Sets the random seed for reproducibility across PyTorch, NumPy, and Python's random module."""
    os.environ['PYTHONHASHSEED'] = str(seed)  # For Python's hash seed
    torch.manual_seed(seed)  # For PyTorch's CPU and CUDA RNGs
    torch.cuda.manual_seed(seed)  # For CUDA devices specifically
    torch.cuda.manual_seed_all(seed) # For all CUDA devices if multiple are used
    np.random.seed(seed)  # For NumPy's random number generator
    random.seed(seed)  # For Python's built-in random module

    # For deterministic algorithms in PyTorch (optional, but recommended for full reproducibility)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Example usage:
set_seed(123) 

class RealNVPCouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim, num, zero_init_last=True):
        super().__init__()
        self.n = num  # used to alternate coupling partitions

        # Scale (s) network
        self.s = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2),
            nn.Tanh()
        )

        # Translation (t) network
        self.t = nn.Sequential(
            nn.Linear(dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim // 2)
        )

        # Initialize weights
        self._init_weights(zero_init_last)

    def _init_weights(self, zero_init_last):
        """Custom weight initialization for RealNVP submodules."""
        for m in list(self.s) + list(self.t):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.02)
                nn.init.constant_(m.bias, 0.0)

        # Optionally zero-init the last layer for stability
        if zero_init_last:
            nn.init.constant_(self.s[-2].weight, 0.0)
            nn.init.constant_(self.s[-2].bias, 0.0)
            nn.init.constant_(self.t[-1].weight, 0.0)
            nn.init.constant_(self.t[-1].bias, 0.0)

    def forward(self, x: torch.Tensor, reverse=False):
        # Split input into two halves
        x1, x2 = x.chunk(2, dim=-1)

        # Alternate which half is used for conditioning
        if self.n % 2 == 1:
            x1, x2 = x2, x1

        s, t = self.s(x1), self.t(x1)

        if not reverse:
            # Forward: x → z (for log-likelihood)
            y1 = x1
            y2 = x2 * torch.exp(s) + t
            log_det_J = s.sum(dim=-1)
        else:
            # Inverse: z → x (for sampling)
            y1 = x1
            y2 = (x2 - t) * torch.exp(-s)
            log_det_J = -s.sum(dim=-1)

        # Recombine
        if self.n % 2 == 1:
            y1, y2 = y2, y1

        y = torch.cat([y1, y2], dim=-1)
        return y, log_det_J


class NormalizingFlow(nn.Module):
    def __init__(self, num_layers:int, dim:int, hidden_dim:int):
        super().__init__()
        self.num_layers = num_layers 
        self.dim = dim 
        self.hidden_dim = hidden_dim
        self.layers = nn.ModuleList([RealNVPCouplingLayer(dim, hidden_dim, n) for n in range(num_layers)])
        self.prior = torch.distributions.normal.Normal(loc=0.0, scale=1.0)  # e.g. N(0, I)

    def forward(self, x):
        log_det_J_total = 0
        h = x
        for layer in self.layers:
            h, log_det_J = layer(h, reverse=False)
            log_det_J_total += log_det_J
        z = h
        z = z.unsqueeze(-1)
        # per-sample log_prob
        log_pz = self.prior.log_prob(z).sum(dim=[1,2])  
        log_px = log_det_J_total + log_pz   # shape [batch_size]

        # expectation over dataset ~ average over batch
        nll = -log_px.mean()
        return z, nll.squeeze() 

    def sample(self, num_samples, seq_len, device:str = "cpu"):
        self.eval()
        z = self.prior.sample((num_samples,seq_len))
        h = z.to(device=device)
        for layer in reversed(self.layers):
            h, _ = layer(h, reverse=True)  # inverse mapping
        x = h
        return x


class CustomDataset(Dataset):
    def __init__(self, X, y, device, classification=True):
        self.X = torch.tensor(X, dtype=torch.float32, device=device)
        if y is not None:
            if classification:
                self.y = torch.tensor(y, dtype=torch.long, device=device)
            else:
                self.y = torch.tensor(y, dtype=torch.float32, device=device)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


# -----------------------
# NF Trainer
# -----------------------
class NFTrainer:
    def __init__(self, model: NormalizingFlow, optimizer, device="cpu", batch_size=32,
                 fp16=False, experiment_name="VAE-Experiment"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16
        self.scaler = GradScaler('cuda', enabled=fp16)
        self.experiment_name = experiment_name

    def _single_pass(self, X):
        self.optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            z, nll = self.model(X)
            
        if self.fp16:
            self.scaler.scale(nll).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            nll.backward()
            self.optimizer.step()

        return z, nll
    
    def _single_nograd_pass(self, X):

        with torch.no_grad():
            z, nll = self.model(X)

        return z, nll
        

    def _log_normalization(self, Z, X, epoch, max_items=8):
        """Log histograms of original vs normalized signals into MLflow"""
        Z = Z[:max_items].detach().cpu().numpy()
        X = X[:max_items].detach().cpu().numpy()

        # Ensure shape is [batch, seq_len]
        if Z.ndim == 3:   # [B, 1, L]
            Z = Z[:, 0, :]
        if X.ndim == 3:
            X = X[:, 0, :]

        fig, axes = plt.subplots(nrows=max_items, ncols=2, figsize=(6, max_items*1.5))
        for i in range(max_items):
            # Left: histogram of the latent vector Z[i]
            axes[i, 0].hist(Z[i].ravel(), bins=30, color="black", alpha=0.7)
            axes[i, 0].set_title(f"Z Histogram")

            axes[i, 1].hist(X[i].ravel(), bins=30, color="black", alpha=0.7)
            axes[i, 1].set_title("X constructed")
            #axes[i, 1].axis("off")

        plt.tight_layout()
        fname = f"X_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close(fig)
        mlflow.log_artifact(fname)
        os.remove(fname)
        
    def _log_samples(self, n_samples=8, seq_len=128, epoch=0):
        """Sample from latent space and log signals to MLflow"""
        self.model.eval()
        with torch.no_grad():
            samples = self.model.sample(n_samples, seq_len, self.device).cpu().numpy()

        fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(6, n_samples*1.5))
        for i in range(n_samples):
            axes[i].plot(samples[i].squeeze(), color="blue")
            #axes[i].axis("off")
        plt.tight_layout()

        fname = f"samples_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close(fig)
        mlflow.log_artifact(fname) 
        os.remove(fname)

    def fit(self, X_train, X_val=None, epochs=20, early_stopping=5):
        # Prepare datasets
        train_loader = DataLoader(CustomDataset(X_train, None, self.device),
                                  sampler=RandomSampler(X_train),
                                  batch_size=self.batch_size,
                                  drop_last=True)
        val_loader = None
        if X_val is not None:
            val_loader = DataLoader(CustomDataset(X_val, None, self.device),
                                    batch_size=self.batch_size,
                                    shuffle=True,
                                    drop_last=False)

        # MLflow logging
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            
                # Model hyperparameters
            mlflow.log_params({
                "dim": self.model.dim,
                "hidden_dims": self.model.hidden_dim,
                "number layers": self.model.num_layers,
                "batch_size": self.batch_size,
                "epochs": epochs
            })
            
            best_val_loss = float("inf")
            patience = early_stopping
            history = {"train_loss": [], "val_loss": []}

            for epoch in range(1, epochs+1):
                # ---- Train ----
                self.model.train()
                train_losses = []
                for X in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                    z, loss= self._single_pass(X)
                    train_losses.append(loss)

                avg_train_loss = sum(train_losses)/len(train_losses)
                history["train_loss"].append(avg_train_loss)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)

                # ---- Validation ----
                avg_val_loss = None
                if val_loader is not None:
                    self.model.eval()
                    val_losses = []
                    with torch.no_grad():
                        for X in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                            z, loss = self._single_nograd_pass(X)
                            val_losses.append(loss.cpu())

                        avg_val_loss = np.mean(val_losses)
                        history["val_loss"].append(avg_val_loss)
                        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                        # Log reconstructions and samples
                        self._log_normalization(z, X, epoch)
                        self._log_samples(n_samples=8, seq_len=X.shape[-1], epoch=epoch)

                        # Save best model
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience = early_stopping
                            torch.save(self.model.state_dict(), "../models/best_nf.pt")
                        else:
                            patience -= 1
                            if patience == 0:
                                print("Early stopping.")
                                break
            mlflow.log_artifact("../models/best_nf.pt")

            
            return history


def load_nf(model_class, checkpoint_path, device="cpu", **model_kwargs) -> NormalizingFlow:
    """
    Load NF model weights from a checkpoint.

    Args:
        model_class: The NF class (e.g., NF)
        checkpoint_path (str): Path to .pth file with state_dict
        device (str): "cpu" or "cuda"
        model_kwargs: Extra kwargs to initialize the model (latent_dim, hidden_dims, etc.)

    Returns:
        model (nn.Module): NF with loaded weights
    """
    # Initialize the model with the same architecture as when saved
    model = model_class(**model_kwargs).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model

