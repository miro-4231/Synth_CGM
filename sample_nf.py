from src.NF_src import NormalizingFlow, load_nf 
from torch.cuda import is_available
from torch import concat, save
from tqdm import tqdm

device = "cuda" if is_available() else "cpu"

model = load_nf(NormalizingFlow, "models\\best_nf.pt", device, num_layers = 16, dim = 128, hidden_dim = 256)

def sample_by_batch_nf(device, num_samples:int, batch_size: int = 1024): 
    
    num_batches = num_samples // batch_size 
    remaining = num_samples % batch_size 
    
    samples = model.sample(remaining, 128, device=device) 
    
    for _ in tqdm(range(num_batches)): 
        
        samples = concat([samples, model.sample(batch_size, 128, device=device)]) 
        
    return samples 

if __name__ == "__main__":
    synth_samples = sample_by_batch_nf(model, device, 67477 ) 
    # Save the tensor to a file named 'my_tensor.pt'
    save(synth_samples.cpu(), 'data\generated\synth_nf.pt')