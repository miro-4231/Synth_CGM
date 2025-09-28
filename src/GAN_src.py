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


#-------+
# Model + 
#-------+

# -----------------------
# Generator
# -----------------------
class Generator1D(nn.Module):
    def __init__(self, z_dim=16, signal_length=128, base_channels=64):
        super().__init__()
        self.z_dim = z_dim
        self.signal_length = signal_length

        # Project latent vector into a feature map
        self.fc = nn.Linear(z_dim, base_channels * (signal_length // 16))

        # Upsample progressively
        self.net = nn.Sequential(
            nn.ConvTranspose1d(base_channels, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.ReLU(True),

            nn.ConvTranspose1d(base_channels // 2, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels // 4),
            nn.ReLU(True),

            nn.ConvTranspose1d(base_channels // 4, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels // 8),
            nn.ReLU(True),

            nn.ConvTranspose1d(base_channels // 8, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)  # (B, base_channels * L/16)
        x = x.view(z.size(0), -1, self.signal_length // 16)  # (B, C, L/16)
        return self.net(x)  # (B, 1, L)


# -----------------------
# Discriminator
# -----------------------
class Discriminator1D(nn.Module):
    def __init__(self, signal_length=128, base_channels=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1, base_channels // 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(base_channels // 8, base_channels // 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels // 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(base_channels // 4, base_channels // 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv1d(base_channels // 2, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base_channels * (signal_length // 16), 1), 
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)  # raw logits (B,1)
    
    
# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

def d_loss_fn(real_logits, fake_logits):
    """
    Discriminator loss: wants real=1, fake=0
    """
    real_labels = torch.ones_like(real_logits)
    fake_labels = torch.zeros_like(fake_logits)
    real_loss = F.binary_cross_entropy_with_logits(real_logits, real_labels)
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, fake_labels)
    return real_loss + fake_loss

def g_loss_fn(fake_logits):
    """
    Generator loss: wants fake=1 (fool the discriminator)
    """
    real_labels = torch.ones_like(fake_logits)
    return F.binary_cross_entropy_with_logits(fake_logits, real_labels)


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
# DCGAN Trainer
# -----------------------
class DCGANTrainer:
    def __init__(self, generator:Generator1D, discriminator:Discriminator1D, g_optimizer, d_optimizer,
                 device="cpu", batch_size=32, fp16=False, experiment_name="DCGAN-Experiment"):
        self.G = generator.to(device)
        self.D = discriminator.to(device)
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.fixed_noize = torch.randn(8, self.G.z_dim)
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16
        self.scaler = GradScaler('cuda', enabled=fp16)
        self.experiment_name = experiment_name

    def _train_step(self, real_data, z_dim):
        real_data = real_data.to(self.device).unsqueeze(1)  # (B,1,L)
        batch_size = real_data.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1, device=self.device)
        fake_labels = torch.zeros(batch_size, 1, device=self.device)

        # =============== Train Discriminator ===============
        z = torch.randn(batch_size, z_dim, device=self.device)
        fake_data = self.G(z)

        self.d_optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            real_pred = self.D(real_data)
            fake_pred = self.D(fake_data.detach())

            d_loss_real = F.binary_cross_entropy_with_logits(real_pred, real_labels)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_pred, fake_labels)
            d_loss = d_loss_real + d_loss_fake

        if self.fp16:
            self.scaler.scale(d_loss).backward()
            self.scaler.step(self.d_optimizer)
        else:
            d_loss.backward()
            self.d_optimizer.step()

        # =============== Train Generator ===============
        self.g_optimizer.zero_grad()
        with autocast(device_type="cuda", dtype=torch.float16, enabled=self.fp16):
            fake_pred = self.D(fake_data)
            g_loss = F.binary_cross_entropy_with_logits(fake_pred, real_labels)

        if self.fp16:
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.g_optimizer)
            self.scaler.update()
        else:
            g_loss.backward()
            self.g_optimizer.step()

        return d_loss.item(), g_loss.item(), fake_data.detach().cpu()

    def _log_samples(self, samples, epoch, n_samples=8):
        samples = samples[:n_samples].squeeze(1)
        samples = self.denormalize(samples).numpy()
        fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(6, n_samples*1.5))
        for i in range(n_samples):
            axes[i].plot(samples[i], color="blue")
        plt.tight_layout()
        fname = f"dcgan_samples_epoch_{epoch}.png"
        plt.savefig(fname)
        plt.close(fig)
        mlflow.log_artifact(fname)
        os.remove(fname)
        
    def normalize(self, X_train):
        self.x_max = X_train.max()
        self.x_min = X_train.min()
        return 2 * (X_train - self.x_min) / (self.x_max - self.x_min) - 1
    
    def denormalize(self, X_train):
        return (X_train + 1) / 2 * (self.x_max - self.x_min) + self.x_min
    
    def fit(self, X_train, z_dim=16, epochs=20, normalize:bool = True):
        train_loader = DataLoader(CustomDataset(X_train, None, self.device),
                                  sampler=RandomSampler(X_train),
                                  batch_size=self.batch_size,
                                  drop_last=True, )
                                  
        if normalize: 
            X_train = self.normalize(X_train)
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            mlflow.log_params({
                "z_dim": z_dim,
                "batch_size": self.batch_size,
                "epochs": epochs
            })

            history = {"d_loss": [], "g_loss": []}

            for epoch in range(1, epochs+1):
                d_losses, g_losses = [], []
                for X in tqdm.tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                    d_loss, g_loss, fake_samples = self._train_step(X, z_dim)
                    d_losses.append(d_loss)
                    g_losses.append(g_loss)

                avg_d = np.mean(d_losses)
                avg_g = np.mean(g_losses)
                history["d_loss"].append(avg_d)
                history["g_loss"].append(avg_g)

                mlflow.log_metric("d_loss", avg_d, step=epoch)
                mlflow.log_metric("g_loss", avg_g, step=epoch)

                self._log_samples(fake_samples, epoch)

                # Save checkpoints
            torch.save(self.G.state_dict(), f"../models/dcgan_G.pt")
            mlflow.log_artifact("../models/dcgan_G.pt")
            torch.save(self.D.state_dict(), f"../models/dcgan_D.pt")
            mlflow.log_artifact("../models/dcgan_D.pt")

            return history
