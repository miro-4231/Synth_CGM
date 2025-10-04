from src.DDPM_src import Unet, load_ddpm 
from torch.cuda import is_available
from torch import concat, save

device = "cuda" if is_available() else "cpu"

model = load_ddpm(Unet, "models\\best_nf.pt", device,     
    dim=8,
    channels=1,
    dim_mults=(1, 2, 4, 8,))

def sample(model:Unet, device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, device) 
    
    for _ in range(num_batches): 
        
        samples = concat([samples, model.sample(batch_size, device)]) 
        
    return samples 

synth_samples = sample(model, device, 67477 ) 

# Save the tensor to a file named 'my_tensor.pt'
save(synth_samples.cpu(), 'data\generated\synth_nf.pt')