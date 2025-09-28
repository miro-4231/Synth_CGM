from src.GAN_src import Generator1D, load_gan 
from torch.cuda import is_available
from torch import concat, save

device = "cuda" if is_available() else "cpu"

model = load_gan(Generator1D, "models\\dcgan_G.pt", device, z_dim=16, signal_length=128)

def sample(model:Generator1D, device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, device) 
    
    for _ in range(num_batches): 
        
        samples = concat([samples, model.sample(batch_size, device)]) 
        
    return samples 

synth_samples = sample(model, device, 67477 ) # number of training instances in training set

# Save the tensor to a file 
save(synth_samples.cpu(), 'data\generated\synth_gan.pt')