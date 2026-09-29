# utils.py
import os, random
import numpy as np
import torch
from dataclasses import dataclass

def init_global_seed(base_seed: int = 13):
    # One-time, at program start
    random.seed(base_seed)
    os.environ["PYTHONHASHSEED"] = str(base_seed)
    np.random.seed(base_seed)
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed_all(base_seed)

    # Determinism knobs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional: make BLAS/tensor ops deterministic
    # os.environ.setdefault("OMP_NUM_THREADS", "1")
    # torch.set_num_threads(1)

@dataclass
class ReproContext:
    np_rng: np.random.Generator
    torch_gen: torch.Generator
    py_rng: random.Random  # for APIs that insist on python's random
    def seed_worker(self, worker_id: int):
        s = self.py_rng.seed() if False else self_seed + worker_id  # placeholder to keep signature

def make_repro_context(seed: int) -> ReproContext:
    np_rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)
    py_rng = random.Random(seed)

    def _seed_worker(worker_id):
        # Seed per DataLoader worker deterministically
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    ctx = ReproContext(np_rng=np_rng, torch_gen=torch_gen, py_rng=py_rng)
    # attach the closure
    ctx.seed_worker = _seed_worker
    return ctx