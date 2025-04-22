import numpy as np
import random
import os
import torch

# 使用百分位数计算阈值
def threshold_q(data, ratio=0.5):
    """
    使用百分位数方法计算阈值
    
    Args:
        data (numpy.ndarray): 输入数据数组
        ratio (float): 通过阈值的数的比例
        
    Returns:
        float: 计算得到的阈值
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