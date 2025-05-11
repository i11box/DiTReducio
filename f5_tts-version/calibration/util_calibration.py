import numpy as np

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