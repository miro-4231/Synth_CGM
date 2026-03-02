from src.VAE_src import VAE, load_vae 
from torch.cuda import is_available
from torch import concat, save

device = "cuda" if is_available() else "cpu"

model:VAE = load_vae(VAE, "models\\best_vaes.pt", device, input_shape=(1, 128), latent_dim=24,)

def sample_by_batch_vae(device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, device) 
    
    for _ in range(num_batches): 
        
        samples = concat([samples, model.sample(batch_size, device)]) 
        
    return samples 

if __name__ == "__main__":
    synth_samples = sample_by_batch_vae(model, device, 67477 ) 
    print(" mean:", synth_samples.mean().item(), " std:", synth_samples.std().item())
    # Save the tensor to a file named 'my_tensor.pt'
    save(synth_samples.cpu(), 'data\generated\synth_vaes.pt')
    