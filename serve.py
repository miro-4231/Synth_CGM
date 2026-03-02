from fastapi import FastAPI
from torch.cuda import is_available
from sample_ddpm import sample_by_batch_ddpm
from sample_vae import sample_by_batch_vae
from sample_nf import sample_by_batch_nf
from sample_gan import sample_by_batch_gan

device = "cuda" if is_available() else "cpu"

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/ddpm/{num_samples}")
def read_item(num_samples: int):
    assert num_samples < 100, "num_samples must be less than 100" 
    samples = sample_by_batch_ddpm(device, num_samples)    
    return samples.tolist() 