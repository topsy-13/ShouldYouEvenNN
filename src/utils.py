import os
import random
import numpy as np
import torch

def set_seed(seed: int = 13) -> None:
    """
    Set random seeds for reproducibility across common modules.
    Works for CPU and GPU (PyTorch).
    """
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Make Python hash-based ops deterministic (env variable)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # print(f"[INFO] Seed set to {seed} for reproducibility.")
