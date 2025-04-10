import numpy as np
from sklearn.cluster import DBSCAN
import random
import os
import torch
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

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

def calculate_sse(data, labels):
    """
    计算聚类的SSE (Sum of Squared Errors)
    
    参数:
    data: 输入数据
    labels: 聚类标签
    
    返回:
    sse: 聚类的SSE值
    """
    unique_labels = np.unique(labels)
    # 排除噪声点（标签为-1的点）
    if -1 in unique_labels:
        unique_labels = unique_labels[unique_labels != -1]
    
    # 如果没有有效聚类，返回无穷大
    if len(unique_labels) == 0:
        return float('inf')
    
    sse = 0.0
    for label in unique_labels:
        # 获取当前聚类的所有点
        cluster_points = data[labels == label]
        # 计算聚类中心
        centroid = np.mean(cluster_points, axis=0)
        # 计算每个点到中心的距离平方和
        cluster_sse = np.sum(np.square(cluster_points - centroid))
        sse += cluster_sse
    
    return sse

def optimize_dbscan_params(data, max_evals=100):
    """
    使用hyperopt优化DBSCAN的参数
    
    参数:
    data: 输入数据
    max_evals: 最大评估次数
    
    返回:
    best_params: 最优参数
    best_labels: 使用最优参数的聚类标签
    best_sse: 最优SSE值
    """
    # 禁用hyperopt的日志输出
    import logging
    hyperopt_logger = logging.getLogger('hyperopt')
    original_level = hyperopt_logger.level
    hyperopt_logger.setLevel(logging.WARNING)  # 或使用logging.ERROR来完全禁止信息
    
    try:
        # 将数据展平为一维数组
        sampled_data = data.flatten().reshape(-1, 1)
        
        # 定义目标函数
        def objective(params):
            eps = params['eps']
            min_samples = int(params['min_samples'])
            
            # 使用DBSCAN进行聚类
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(sampled_data)
            
            # 计算聚类数量（不包括噪声点）
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # 如果产生了多个聚类，返回一个很大的惩罚值
            if n_clusters > 1:
                return {'loss': 1e10, 'status': STATUS_OK, 'n_clusters': n_clusters}
            
            # 如果没有聚类（所有点都是噪声点），返回一个很大的惩罚值
            if n_clusters == 0:
                return {'loss': 1e10, 'status': STATUS_OK, 'n_clusters': n_clusters}
            
            # 计算SSE
            sse = calculate_sse(sampled_data, labels)
            
            # 计算噪声点比例
            noise_ratio = np.sum(labels == -1) / len(labels)
            
            # 添加一个小的惩罚项，使噪声点比例不要太高
            penalty = noise_ratio * 0.1 * sse
            
            return {
                'loss': sse + penalty,
                'status': STATUS_OK,
                'eps': eps,
                'min_samples': min_samples,
                'sse': sse,
                'noise_ratio': noise_ratio,
                'n_clusters': n_clusters
            }
        
        # 定义参数搜索空间
        space = {
            'eps': hp.uniform('eps', 0.01, 1.0),
            'min_samples': hp.quniform('min_samples', 2, 50, 1)
        }
        
        # 创建Trials对象来存储每次评估的结果
        trials = Trials()
        
        # 使用TPE算法进行优化
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            trials=trials,
            rstate=np.random.default_rng(42),  # 设置随机种子以确保可重复性
            verbose=0  # 禁用fmin的详细输出
        )
        
        # 获取最优参数
        best_eps = best['eps']
        best_min_samples = int(best['min_samples'])
        
        print(f"\n优化结果:")
        print(f"最优eps: {best_eps:.4f}")
        print(f"最优min_samples: {best_min_samples}")
        
        # 使用最优参数在完整数据集上运行DBSCAN
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        flat_data = sampled_data
        best_labels = dbscan.fit_predict(flat_data)
        
        # 计算最终的SSE
        best_sse = calculate_sse(flat_data, best_labels)
        
        # 统计聚类结果
        n_noise = list(best_labels).count(-1)
        print(f"最终聚类结果: {n_noise} 个噪声点 ({n_noise/len(best_labels)*100:.2f}%)")
        print(f"最终SSE: {best_sse:.4f}")
        
        # 返回最优参数和结果
        best_params = {
            'eps': best_eps,
            'min_samples': best_min_samples
        }
        
        return best_params, best_labels, best_sse
    
    finally:
        # 恢复原来的日志级别
        hyperopt_logger.setLevel(original_level)

# 使用DBSCAN方法计算阈值
def threshold_dbscan(data):
    # 将数据展平为一维数组
    flat_data = data.flatten().reshape(-1, 1)
    
    best_params, labels, best_sse = optimize_dbscan_params(data, max_evals=200)
    print(f"最佳sse: {best_sse:.4f}")
    
    eps,min_samples = best_params['eps'],best_params['min_samples']
    
    # 如果有噪声点（标签为-1），则找到噪声点的最小值作为阈值
    if -1 in labels:
        noise_points = flat_data[labels == -1]
        threshold = np.min(noise_points)
        return float(threshold)
    
    # 如果没有噪声点，则尝试减小eps直到找到噪声点
    for _ in range(10):  # 最多尝试10次
        eps *= 0.8  # 减小eps
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(flat_data)
        labels = clustering.labels_
        if -1 in labels:
            noise_points = flat_data[labels == -1]
            threshold = np.min(noise_points)
            return float(threshold)
    
    # 如果仍然没有找到噪声点，则使用数据的90%分位数作为阈值
    print("没有找到噪声点,使用90%分位数作为阈值")
    return float(np.percentile(flat_data, 90))

def seed_everything(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False