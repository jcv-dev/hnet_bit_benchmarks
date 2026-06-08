import torch
import os

checkpoint_path = "runs/sanity_hier/checkpoint_final.pt"

if os.path.exists(checkpoint_path):
    # Check file size
    file_size_gb = os.path.getsize(checkpoint_path) / (1024**3)
    print(f"Checkpoint file size: {file_size_gb:.2f} GB")
    
    # Load and inspect
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Calculate weight memory
    weight_bytes = 0
    for k, v in ckpt['model_state_dict'].items():
        weight_bytes += v.element_size() * v.numel()
    
    weight_mb = weight_bytes / (1024**2)
    print(f"Model weights only: {weight_mb:.2f} MB ({weight_bytes / (1024**3):.3f} GB)")
    
    # Optimizer is bigger (fp32 states)
    if 'optimizer_state_dict' in ckpt:
        opt_size = sum(v.element_size() * v.numel() 
                      for state in ckpt['optimizer_state_dict']['state'].values()
                      for v in state.values() if isinstance(v, torch.Tensor))
        opt_gb = opt_size / (1024**3)
        print(f"Optimizer states: {opt_gb:.2f} GB")