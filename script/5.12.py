import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置图表样式
sns.set(style='whitegrid', font_scale=1.5)  # 增大字体比例
plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 17  # x轴刻度标签字体大小
plt.rcParams['ytick.labelsize'] = 17  # y轴刻度标签字体大小
plt.rcParams['legend.fontsize'] = 16  # 图例字体大小
plt.rcParams['axes.titlesize'] = 18  # 标题字体大小
plt.rcParams['lines.linewidth'] = 2.5  # 线宽
plt.rcParams['lines.markersize'] = 10  # 标记大小

# 创建数据
thresholds = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6']

# F5-TTS 数据
f5_tts_attention = [21.878, 19.944, 17.182, 15.134, 13.288, 11.628, 10.328]
f5_tts_feedforward = [11.47, 9.846, 8.188, 6.992, 5.944, 5.176, 4.608]

# MegaTTS 3 数据
mega_tts_attention = [12.855, 12.768, 12.372, 10.513, 7.533, 5.785, 5.205]
mega_tts_feedforward = [63.106, 61.046, 53.087, 41.291, 27.482, 19.895, 16.833]

# 创建图表
fig, ax = plt.subplots(figsize=(12, 8))

# 设置颜色和标记
palette = sns.color_palette('Set2')  # 使用Set2调色板

# 绘制四条线
ax.plot(thresholds, f5_tts_attention, label='F5-TTS Attention', color=palette[0], marker='o', linewidth=2, markersize=8)
ax.plot(thresholds, f5_tts_feedforward, label='F5-TTS Feed-Forward', color=palette[0], marker='s', linewidth=2, markersize=8, linestyle='--')
ax.plot(thresholds, mega_tts_attention, label='MegaTTS 3 Attention', color=palette[1], marker='^', linewidth=2, markersize=8)
ax.plot(thresholds, mega_tts_feedforward, label='MegaTTS 3 Feed-Forward', color=palette[1], marker='d', linewidth=2, markersize=8, linestyle='--')

# 设置标签
ax.set_xlabel('Compression Threshold', labelpad=15, fontsize=20)
ax.set_ylabel('Module Latency', labelpad=15, fontsize=20)

# 添加网格线和图例
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='best', frameon=True)

# 设置x轴刻度
ax.set_xticks(range(7))
ax.set_xticklabels(thresholds)
ax.tick_params(axis='both', which='major', pad=10)

# 调整布局
plt.tight_layout(pad=2.5)

# 保存图表
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')

# 显示图表
plt.show()
