from fastapi import FastAPI, HTTPException
from torch.cuda import is_available
from sample_ddpm import sample_by_batch_ddpm
from sample_vae import sample_by_batch_vae
from sample_nf import sample_by_batch_nf
from sample_gan import sample_by_batch_gan

device = "cuda" if is_available() else "cpu"

app = FastAPI()

MAX_SAMPLES = 100

def validate_num_samples(num_samples: int):
    if num_samples > MAX_SAMPLES:
        raise HTTPException(status_code=400, detail=f"num_samples must be <= {MAX_SAMPLES}")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ddpm/{num_samples}")
def sample_ddpm(num_samples: int):
    validate_num_samples(num_samples)
    samples = sample_by_batch_ddpm(device, num_samples)
    return samples.tolist()

@app.get("/vae/{num_samples}")
def sample_vae(num_samples: int):
    validate_num_samples(num_samples)
    samples = sample_by_batch_vae(device, num_samples)
    return samples.tolist()

@app.get("/nf/{num_samples}")
def sample_nf(num_samples: int):
    validate_num_samples(num_samples)
    samples = sample_by_batch_nf(device, num_samples)
    return samples.tolist()

@app.get("/gan/{num_samples}")
def sample_gan(num_samples: int):
    validate_num_samples(num_samples)
    samples = sample_by_batch_gan(device, num_samples)
    return samples.tolist()