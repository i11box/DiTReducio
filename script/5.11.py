import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 创建数据框
data = {
    'Model': [
        'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS',
        'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS',
        'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE',
        'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS',
        'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS',
        'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE',
        'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS',
        'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS',
        'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE',
        'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS', 'w/o TS',
        'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS', 'w/o BS',
        'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE', 'w/o PRE'
    ],
    'Metric': [
        'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o',
        'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o',
        'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o', 'SIM-o',
        'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)',
        'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)',
        'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)', 'WER (%)',
        'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF',
        'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF',
        'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF', 'RTF',
        'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)',
        'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)',
        'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)', 'Ops Ratio (%)'
    ],
    'Threshold': [
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
        'T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'
    ],
    'Value': [
        0.640, 0.636, 0.628, 0.628, 0.615, 0.635, 0.549,  # w/o TS SIM-o
        0.640, 0.638, 0.607, 0.614, 0.609, 0.567, 0.476,  # w/o BS SIM-o
        0.640, 0.639, 0.627, 0.600, 0.557, 0.542, 0.518,  # w/o PRE SIM-o
        2.636, 2.641, 2.743, 2.595, 2.685, 2.566, 2.836,  # w/o TS WER
        2.636, 2.628, 2.533, 3.306, 4.569, 8.556, 23.059,  # w/o BS WER
        2.734, 2.572, 2.643, 2.706, 2.988, 3.550, 6.509,  # w/o PRE WER
        0.178, 0.171, 0.168, 0.161, 0.160, 0.160, 0.159,  # w/o TS RTF
        0.178, 0.158, 0.133, 0.120, 0.109, 0.097, 0.094,  # w/o BS RTF
        0.178, 0.167, 0.152, 0.138, 0.126, 0.119, 0.113,  # w/o PRE RTF
        100.00, 85.63, 74.78, 68.26, 63.93, 61.51, 58.58,  # w/o TS Ops Ratio
        100.00, 77.70, 58.24, 47.16, 38.07, 30.40, 25.99,  # w/o BS Ops Ratio
        100.00, 75.21, 60.09, 48.86, 39.84, 34.38, 29.47   # w/o PRE Ops Ratio
    ]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 设置图表样式
sns.set(style='whitegrid', font_scale=1.5)  # 增大字体比例
plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 17  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 17  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 16  # 图例字体大小
plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['lines.linewidth'] = 2.5  # 线宽
plt.rcParams['lines.markersize'] = 10  # 标记大小

# 创建2x2子图布局
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
axes = axes.flatten()

# 设置颜色和标记 - 使用seaborn调色板
palette = sns.color_palette('Set2')  # 使用colorblind调色板
colors = {'w/o TS': palette[0], 'w/o BS': palette[1], 'w/o PRE': palette[2]}
markers = {'w/o TS': 'o', 'w/o BS': 's', 'w/o PRE': '^'}

# 定义指标顺序
metrics = ['SIM-o', 'WER (%)', 'RTF', 'Ops Ratio (%)']

# 绘制每个指标的子图
for i, metric in enumerate(metrics):
    ax = axes[i]
    metric_data = df[df['Metric'] == metric]
    
    # 为每个模型绘制折线
    for model in ['w/o TS', 'w/o BS', 'w/o PRE']:
        model_data = metric_data[metric_data['Model'] == model]
        ax.plot(model_data['Threshold'], model_data['Value'], 
                label=model, color=colors[model], marker=markers[model], 
                linewidth=2, markersize=8)
    
    # 设置标签（不显示标题）
    ax.set_xlabel('Compression Threshold', labelpad = 15, fontsize = 20)
    ax.set_ylabel(metric, labelpad = 15, fontsize = 20)
    
    # 添加网格线和图例
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best', frameon=True)
    
    # 设置x轴刻度
    ax.set_xticks(range(7))
    ax.set_xticklabels(['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6'])
    ax.tick_params(axis='both', which='major', pad=10)
    
    # 对于WER指标，使用线性坐标轴并调整范围
    if metric == 'WER (%)':
        ax.set_ylim(2.0, 7.0)
        

# 调整布局 - 不显示总标题
plt.tight_layout(pad=2.5)  # 增加子图之间的间距

# 保存图表
plt.savefig('ablation.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
