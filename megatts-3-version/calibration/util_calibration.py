import numpy as np
import random
import os
import torch

def threshold_q(data, ratio=0.5):
    """
    Calculate threshold using the percentile method.

    Args:
        data (numpy.ndarray): Input data array
        ratio (float): The proportion of data points that should be below the threshold

    Returns:
        float: The calculated threshold value
    """
    return float(np.percentile(data, (1 - ratio) * 100))

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False