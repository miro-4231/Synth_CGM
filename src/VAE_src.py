from typing import Tuple, List, Any 
import torch
import numpy as np
import random
import os
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import mlflow
import tqdm
import math

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


def conv1d_output_length(L_in, kernel_size, stride=2, padding=1, dilation=1):
    """
    Compute the output length of a 1D convolution layer.

    Parameters
    ----------
    L_in : int
        Input length (sequence length).
    kernel_size : int
        Size of the convolution kernel.
    stride : int, default=1
        Convolution stride.
    padding : int, default=0
        Padding on both sides.
    dilation : int, default=1
        Dilation factor.

    Returns
    -------
    L_out : int
        Output length after the convolution.
    """
    return math.floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride + 1)


################################################################
#################### Model Architecture ########################
################################################################


class ConvBlock(nn.Module): # DCGAN inspired
    def __init__(self, in_channels, out_channels, kernel_size, down=True, use_bn=True):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, 2, 1, bias=False),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity(),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 2, 1, 1, bias=False),
                nn.BatchNorm1d(out_channels) if use_bn else nn.Identity(),
                nn.ReLU(inplace=True)
            )
    def forward(self, x):
        return self.block(x)



class VAE(nn.Module): 
    
    def __init__(self, input_shape : Tuple[int, int], latent_dim: int , hidden_dims:List[int] = None, kernel_sizes:List[int] = None):
        super(VAE, self).__init__( )
        
        if hidden_dims is None : 
            hidden_dims = [16, 32, 64]
        
        if kernel_sizes is None : 
            kernel_sizes = [5, 4, 3]
            
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.input_shape = input_shape
        self.kernel_sizes = kernel_sizes
        
        # build encoder    
        blocks = []
        in_channels = input_shape[0]
        L_in = input_shape[1]
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes): 
            
            blocks.append(ConvBlock(in_channels, hidden_dim, kernel_size))
            in_channels = hidden_dim
            
            L_out = conv1d_output_length(L_in, kernel_size)
            L_in = L_out
            
        self.encoder = nn.Sequential(*blocks)
            
        self.mu = nn.Linear(L_in * hidden_dims[-1], latent_dim) 
        self.log_var = nn.Linear(L_in * hidden_dims[-1], latent_dim) 
        
        kernel_sizes.reverse()
        hidden_dims.reverse() 
        
        self.decoder_input = nn.Linear(latent_dim, L_in * hidden_dims[0])
        
        # build decoder    
        blocks = []
        in_channels = hidden_dims[0]
        L_out = input_shape[1]
        for layer in range(len(hidden_dims) - 1): 
            
            blocks.append(ConvBlock(hidden_dims[layer], hidden_dims[layer+1], kernel_sizes[layer], down=False))
            
        self.decoder = nn.Sequential(*blocks) 
        
        self.final_layer = nn.Sequential(
            ConvBlock(hidden_dims[-1], hidden_dims[-1], kernel_size=kernel_sizes[-1], down=False), 
            nn.Conv1d(hidden_dims[-1], input_shape[0], 3, padding=0), 
        )
            
    def reparametrize(self, mu, log_var): 
        
        std = torch.exp(0.5 * log_var) 
        
        eps = torch.rand_like(mu) 
        
        return mu + eps * std 
    
    def encode(self, x: Tensor) -> Tuple[Tensor]: 
        
        encoded = self.encoder(x) 
        
        B, _, _ = encoded.shape 
        
        encoded = encoded.reshape(B, -1)
        
        mu = self.mu(encoded)
        log_var = self.log_var(encoded) 
        
        return mu, log_var
    
    def decode(self, z: Tensor) -> Tuple[Tensor]:

        decoder_input = self.decoder_input(z) 
        
        B, _= decoder_input.shape 
        
        decoder_input = decoder_input.reshape(B, self.hidden_dims[0], -1)
        
        decoded = self.decoder(decoder_input) 

        reconstructed = self.final_layer(decoded)  
        
        reconstructed = reconstructed[:, :, :self.input_shape[1]]
        
        return reconstructed
    
    def forward(self, x: Tensor) -> Tuple[Tensor]: 
        
        mu, log_var = self.encode(x) 

        z = self.reparametrize(mu, log_var) 
        
        reconstructed = self.decode(z)
        
        return reconstructed, mu, log_var 
    
    def sample(self, num_samples, device): 
        
        with torch.no_grad(): 
            
            z = torch.randn(num_samples, self.latent_dim, device=device) 
            
            samples = self.decode(z) 
            
        return samples 
        
def vae_loss(x, recon, mu, log_var, beta:int = 1):
    # Flatten to [B, -1] if needed
    recon_loss = torch.nn.functional.mse_loss(
        recon, x, reduction='sum'
    ) / x.size(0)   # normalize by batch

    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) 
    kl_loss /= x.size(0)  # normalize by batch

    return recon_loss + beta * kl_loss, recon_loss, kl_loss



#--------------
# Data Loading 
#--------------

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
# VAE Trainer
# -----------------------
class VAETrainer:
    def __init__(self, model: VAE, optimizer, beta:int, device="cpu", batch_size=32,
                 fp16=False, experiment_name="VAE-Experiment"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16
        self.scaler = GradScaler('cuda', enabled=fp16)
        self.experiment_name = experiment_name
        self.beta = beta

    def _single_pass(self, X, train=True):
        self.optimizer.zero_grad()
        X = X.to(self.device).unsqueeze(1)  # (B,1,L)

        with autocast(device_type='cuda', dtype=torch.float16):
            recon, mu, log_var = self.model(X)
            loss, recon_loss, kl = vae_loss(recon, X, mu, log_var, self.beta)

        if train:
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return (loss.detach().cpu().item(),
                recon.detach().cpu(),
                recon_loss.detach().cpu().item(),
                kl.detach().cpu().item())
        

    def _log_reconstructions(self, X, recon, epoch, max_items=8):
        """Log original vs reconstructed signals into MLflow"""
        X = X[:max_items].detach().cpu().numpy()
        recon = recon[:max_items].detach().cpu().numpy()

        # Ensure shape is [batch, seq_len]
        if X.ndim == 3:   # [B, 1, L]
            X = X[:, 0, :]
        if recon.ndim == 3:
            recon = recon[:, 0, :]

        fig, axes = plt.subplots(nrows=max_items, ncols=2, figsize=(6, max_items*1.5))
        for i in range(max_items):
            axes[i, 0].plot(X[i], color="black")
            if i == 0:
                axes[i, 0].set_title("Original")
            #axes[i, 0].axis("off")

            axes[i, 1].plot(recon[i], color="red")
            if i == 0:
                axes[i, 1].set_title("Reconstruction")
            #axes[i, 1].axis("off")

        plt.tight_layout()
        fname = f"recon_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close(fig)
        mlflow.log_artifact(fname)
        os.remove(fname)
        
    def _log_samples(self, n_samples=8, seq_len=128, epoch=0):
        """Sample from latent space and log signals to MLflow"""
        self.model.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.model.latent_dim, device=self.device)
            samples = self.model.decode(z).cpu().numpy()

        fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(6, n_samples*1.5))
        for i in range(n_samples):
            axes[i].plot(samples[i,0,:], color="blue")
            #axes[i].axis("off")
        plt.tight_layout()

        fname = f"samples_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close(fig)
        mlflow.log_artifact(fname) 
        os.remove(fname)

    def fit(self, X_train, X_val=None, epochs=20, early_stopping=5, early_stoping_criteria:str="loss"):
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
                "latent_dim": self.model.latent_dim,
                "hidden_dims": self.model.hidden_dims,
                "kernel_sizes": self.model.kernel_sizes,
                "beta": self.beta,
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
                recon_losses = []
                kl_losses = []
                for X in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                    loss, _, recon_loss, kl_loss = self._single_pass(X, train=True)
                    train_losses.append(loss)
                    recon_losses.append(recon_loss)
                    kl_losses.append(kl_loss)

                avg_train_loss = np.mean(train_losses)
                avg_train_recon_loss = np.mean(recon_losses)
                avg_train_kl_loss = np.mean(kl_losses)
                history["train_loss"].append(avg_train_loss)
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_recon_loss", avg_train_recon_loss, step=epoch)
                mlflow.log_metric("train_kl_loss", avg_train_kl_loss, step=epoch)

                # ---- Validation ----
                avg_val_loss = None
                if val_loader is not None:
                    self.model.eval()
                    val_losses = []
                    recon_losses = []
                    kl_losses = []
                    with torch.no_grad():
                        for X in tqdm.tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                            loss, recon, recon_loss, kl_loss = self._single_pass(X, train=False)
                            val_losses.append(loss)
                            recon_losses.append(recon_loss)
                            kl_losses.append(kl_loss)

                        avg_val_loss = np.mean(val_losses)
                        avg_recon_loss = np.mean(recon_losses)
                        avg_kl_loss = np.mean(kl_losses)
                        history["val_loss"].append(avg_val_loss)
                        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                        mlflow.log_metric("val_recon_loss", avg_recon_loss, step=epoch)
                        mlflow.log_metric("val_kl_loss", avg_kl_loss, step=epoch)

                        # Log reconstructions and samples
                        self._log_reconstructions(X, recon, epoch, max_items=5)
                        self._log_samples(n_samples=10, seq_len=X.shape[-1], epoch=epoch)
                        if early_stoping_criteria == "loss":
                            monitor_value = avg_val_loss
                        elif early_stoping_criteria == "recon_loss":
                            monitor_value = avg_recon_loss
                        elif early_stoping_criteria == "kl_loss":
                            monitor_value = avg_kl_loss
                        else:
                            print("Warning: Unknown early stopping criteria. Defaulting to 'loss'.")
                            monitor_value = avg_val_loss
                        # Save best model
                        if monitor_value < best_val_loss:
                            best_val_loss = monitor_value
                            patience = early_stopping
                            torch.save(self.model.state_dict(), "../models/best_vae.pt")
                        else:
                            patience -= 1
                            if patience == 0:
                                print("Early stopping.")
                                break
            mlflow.log_artifact("../models/best_vae.pt")

            
            return history

# Wrap loss so it matches (y_pred, y)
def loss_wrapper(y_pred, y):
    recon, mu, log_var = y_pred
    return vae_loss(recon, y, mu, log_var)


def load_vae(model_class, checkpoint_path, device="cpu", **model_kwargs) -> VAE:
    """
    Load VAE model weights from a checkpoint.

    Args:
        model_class: The VAE class (e.g., VAE)
        checkpoint_path (str): Path to .pth file with state_dict
        device (str): "cpu" or "cuda"
        model_kwargs: Extra kwargs to initialize the model (latent_dim, hidden_dims, etc.)

    Returns:
        model (nn.Module): VAE with loaded weights
    """
    # Initialize the model with the same architecture as when saved
    model = model_class(**model_kwargs).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model