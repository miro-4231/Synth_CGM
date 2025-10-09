from src.DDPM_src import Unet, load_ddpm, sample
from torch.cuda import is_available
from torch import concat, save
from tqdm import tqdm

device = "cuda" if is_available() else "cpu"

MIN = 40
MAX = 400 

function_reverse = lambda t: (t / 2 + 0.5) * (MAX - MIN) + MIN

model = load_ddpm(Unet, "models\\best_ddpm.pt", device,     
    dim=8,
    channels=1,
    dim_mults=(1, 2, 4, 8,),)

def sample_by_batch(model:Unet, device, num_samples:int, batch_size: int = 1024, seq_len:int = 128, channels:int = 1): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = sample(model, 128, remaining, 1)[-1] 
    print(samples.shape)
    for _ in tqdm(range(num_batches)): 
        
        samples = concat([samples, sample(model, seq_len, batch_size, channels)[-1] ]) 
    samples = function_reverse(samples)    
    return samples 

synth_samples = sample_by_batch(model, device, 67477 ) 

# Save the tensor to a file named 'my_tensor.pt'
save(synth_samples.cpu(), 'data\generated\synth_ddpm.pt')