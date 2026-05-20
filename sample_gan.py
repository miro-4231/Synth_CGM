from sample_ddpm import PH
from src.GAN_src import Generator1D, load_gan 
from torch.cuda import is_available
from torch import concat, save

device = "cuda" if is_available() else "cpu"

PH = 8

HypoTreshold:int = 70
AdverseEventOnly:bool = True

model = load_gan(Generator1D, "models\\dcgan_G.pt", device, z_dim=16, signal_length=128)

def sample_by_batch_gan(device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, device) 
    
    for _ in range(num_batches): 
        
        samples = concat([samples, model.sample(batch_size, device)]) 
        
    return samples 

def sample_adverse_gan(device, num_samples:int, batch_size: int = 1024, seq_len:int = 128, channels:int = 1): 

    
    samples = model.sample( batch_size, device)
    samples = samples[(samples[:,:, -PH:] < HypoTreshold).any(dim=2)]
    
    while len(samples) < num_samples:
        new_samples = model.sample(batch_size, device)
        new_samples = new_samples[(new_samples[:,:, -PH:] < HypoTreshold).any(dim=2)]
        samples = concat([samples, new_samples])

    return samples


if __name__ == "__main__":
    if AdverseEventOnly:
        synth_samples = sample_adverse_gan(device, 5000 )

        save(synth_samples.cpu(), 'data\generated\synthAug_gan.pt')
    else:
        # number of training instances in training set
        synth_samples = sample_by_batch_gan(device, 67477 ) 
        # Save the tensor to a file 
        save(synth_samples.cpu(), 'data\generated\synth_gan.pt')