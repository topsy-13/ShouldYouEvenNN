import os, random
import numpy as np
import torch

def set_seed(seed: int = 13):
    """
    Set seeds for reproducibility across Python, NumPy, and PyTorch.
    Returns a torch.Generator and worker_init_fn for DataLoader use.
    """
    # Python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cuDNN determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Generator for DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    # Worker seeding function
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    return g, seed_worker
