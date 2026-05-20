from src.VAE_src import VAE, load_vae 
from torch.cuda import is_available
from torch import concat, save

PH = 8

HypoTreshold:int = 70
AdverseEventOnly:bool = True

device = "cuda" if is_available() else "cpu"

model:VAE = load_vae(VAE, "models\\best_vae.pt", device, input_shape=(1, 128), latent_dim=24,)

def sample_by_batch_vae(device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, device) 
    
    for _ in range(num_batches): 
        
        samples = concat([samples, model.sample(batch_size, device)]) 
        
    return samples 

def sample_adverse_vae(device, num_samples:int, batch_size: int = 1024): 

    
    samples = model.sample( batch_size,device)
    samples = samples[(samples[:,:, -PH:] < HypoTreshold).any(dim=2)]
    
    while len(samples) < num_samples:
        new_samples = model.sample(batch_size, device)
        new_samples = new_samples[(new_samples[:,:, -PH:] < HypoTreshold).any(dim=2)]
        samples = concat([samples, new_samples])

    return samples

if __name__ == "__main__":
    if AdverseEventOnly:
        synth_samples = sample_adverse_vae(device, 5000 )

        save(synth_samples.cpu(), 'data\generated\synthAug_vae.pt')
    else:
        synth_samples = sample_by_batch_vae(device, 67477 ) 

        save(synth_samples.cpu(), 'data\generated\synth_vae.pt')
    