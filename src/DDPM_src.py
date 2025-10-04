import torch
import numpy as np
import random
import os
from matplotlib import pyplot as plt
import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.optim as optim 
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.amp import GradScaler, autocast
import mlflow


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

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (T p) -> b (c p) T", p=2),
        nn.Conv1d(dim * 2, default(dim_out, dim), 1),
    )
    
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o i k -> o 1 1 ", "mean")
        var = reduce(weight, "o i k -> o 1 1 ", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h t d -> b (h d) t", t=t)
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) t -> b h c t", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c t -> b (h c) t", h=self.heads, t=t)
        return self.to_out(out)



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.dim_mults = dim_mults 
        self.init_dim = init_dim 
        self.grps = resnet_block_groups
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start



timesteps = 300

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


from torchvision.transforms import Compose, ToTensor, Lambda
# To improve further ####################################################################################################################################################################################################
# for OhioT1DM
MIN = 40
MAX = 400
#########################################################################################################################################################################################################################
transform = Compose([
    ToTensor(), 
    Lambda(lambda t: (((t - MIN) / (MAX - MIN)) - 0.5) * 2),
    Lambda(lambda t: t.permute(1, 0, 2))
])


reverse_transform = Compose([
    Lambda(lambda t: (t / 2 + 0.5) * (MAX - MIN) + MIN),
    Lambda(lambda t: t.numpy().astype(np.uint32)),
])


def get_noisy_signal(x_start, t):
  # add noise
  x_noisy = q_sample(x_start, t=t)
  
  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image


def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0]

    # start from noise
    signal = torch.randn(shape, device=device)
    signals = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        signal = p_sample(model, signal, t, i)
        signals.append(signal.cpu())

    return torch.stack(signals)


@torch.no_grad()
def sample(model, signal_length, batch_size=16, channels=1):
    """
    Sample 1D signals from the diffusion model.

    Args:
        model: The diffusion model
        signal_length (int): The length of each 1D signal
        batch_size (int): Number of signals to generate
        channels (int): Number of signal channels

    Returns:
        List of signal tensors (each of shape [B, C, T]) for all time steps
    """
    return p_sample_loop(model, shape=(batch_size, channels, signal_length))



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
class DDPMTrainer:
    def __init__(self, model: Unet, optimizer, device="cpu", batch_size=32,
                 fp16=False, experiment_name="VAE-Experiment"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.batch_size = batch_size
        self.fp16 = fp16
        self.scaler = GradScaler('cuda', enabled=fp16)
        self.experiment_name = experiment_name

    def _single_pass(self, batch, train=True):
        self.optimizer.zero_grad()
        batch_size = batch.shape[0]
        batch = batch.to(self.device)
        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=self.device).long()

        with autocast(device_type='cuda', dtype=torch.float16):
            loss = p_losses(self.model, batch, t, loss_type="huber")

        if train:
            if self.fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        return loss.detach().cpu().item()
        
        
    def _log_samples(self, n_samples=8, seq_len=128, epoch=0):
        """Sample from latent space and log signals to MLflow"""
        self.model.eval()
        samples = sample(self.model, signal_length=seq_len, batch_size=n_samples, channels=self.model.channels)
        
        samples = samples[-1,:,:,:]
        num_channels = samples.shape[1]
        samples = reverse_transform(samples)
        
        fig, axes = plt.subplots(nrows=n_samples, ncols=1, figsize=(6, n_samples*1.5))
        for i in range(n_samples):
            for channel in range(num_channels):
                axes[i].plot(samples[i,channel,:], color="blue")
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
                "channels": self.model.channels,
                "dim mults": self.model.dim_mults,
                "kernel_sizes": self.model.init_dim,
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
                for X in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
                    loss = self._single_pass(X, train=True)
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
                        for X in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                            loss = self._single_pass(X, train=False)
                            val_losses.append(loss)

                        avg_val_loss = np.mean(val_losses)
                        history["val_loss"].append(avg_val_loss)
                        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)

                        # Log and samples
                        self._log_samples(n_samples=8, seq_len=X.shape[-1], epoch=epoch)

                        # Save best model
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            patience = early_stopping
                            torch.save(self.model.state_dict(), "../models/best_ddpm.pt")
                        else:
                            patience -= 1
                            if patience == 0:
                                print("Early stopping.")
                                break
            mlflow.log_artifact("../models/best_ddpm.pt")

            
            return history

def load_ddpm(model_class, checkpoint_path, device="cpu", **model_kwargs):
    """
    Load DDPM model weights from a checkpoint.

    Args:
        model_class: The DDPM class (e.g., DDPM)
        checkpoint_path (str): Path to .pth file with state_dict
        device (str): "cpu" or "cuda"
        model_kwargs: Extra kwargs to initialize the model (latent_dim, hidden_dims, etc.)

    Returns:
        model (nn.Module): DDPM with loaded weights
    """
    # Initialize the model with the same architecture as when saved
    model = model_class(**model_kwargs).to(device)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model
