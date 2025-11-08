import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 设置中文字体和seaborn样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

# 读取数据
df = pd.read_csv('results/nsga2_pareto_front.csv')

# 检查数据
print("数据前几行：")
print(df.head())
print(f"\n数据形状：{df.shape}")
print(f"列名：{df.columns.tolist()}")

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制散点图
scatter = plt.scatter(df['Makespan'], df['Total_Workload'], 
                     c=df['Solution_ID'], cmap='viridis', 
                     s=80, alpha=0.8, edgecolors='black', linewidth=0.5)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Solution ID', fontsize=12)

# 设置坐标轴标签和标题
plt.xlabel('Makespan', fontsize=14, fontweight='bold')
plt.ylabel('Total Workload', fontsize=14, fontweight='bold')
plt.title('NSGA-II Pareto Front: Makespan vs Total Workload', 
          fontsize=16, fontweight='bold', pad=20)

# 添加网格
plt.grid(True, alpha=0.3)

# 美化坐标轴
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_linewidth(0.5)
plt.gca().spines['bottom'].set_linewidth(0.5)

# 添加一些统计信息
makespan_min, makespan_max = df['Makespan'].min(), df['Makespan'].max()
workload_min, workload_max = df['Total_Workload'].min(), df['Total_Workload'].max()

# 在图上添加统计信息框
stats_text = f'Pareto Solutions: {len(df)}\nMakespan: [{makespan_min:.1f}, {makespan_max:.1f}]\nWorkload: [{workload_min:.1f}, {workload_max:.1f}]'
plt.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top', fontsize=10)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 可选：保存图形
plt.savefig('results/pareto_front_plot.png')
print("图形已保存为 'results/pareto_front_plot.png'")

print("\n绘图完成！")