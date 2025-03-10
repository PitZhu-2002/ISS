import math

from statsmodels.stats.libqsturng import qsturng

# 参数设置
alpha = 0.1  # 显著性水平
k = 12       # 分组数
df = float('inf')  # 自由度

# 计算临界值
q_alpha = qsturng(1 - alpha, k, df) / math.sqrt(2)
print(f"Studentized Range Critical Value (q_alpha): {q_alpha}")