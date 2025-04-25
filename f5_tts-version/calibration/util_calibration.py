import numpy as np
import random
import os
import torch

def threshold_q(data, ratio=0.5):
    """
    top q% threshold
    
    Args:
        data (numpy.ndarray):
        ratio (float)
        
    Returns:
        float: threshold value
    """
    return float(np.percentile(data, (1-ratio) * 100))

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False