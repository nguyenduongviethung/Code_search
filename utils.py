import json
import random
import numpy as np
import torch

def load_json(path: str):
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)

def set_seed(seed: int=42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)