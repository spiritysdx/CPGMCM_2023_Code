import pandas as pd
import itertools

# 示例数据
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 3, 4], 'C': [9, 10, 11, 12], 'D': [13, 14, 15, 16]})

# 示例变量列表
variables = ['A', 'B', 'C', 'D']

# 生成变量的所有组合
combinations = []
for r in range(2, len(variables)+1):
    combinations.extend(itertools.combinations(variables, r))

# 创建空DataFrame存储结果
result_df = pd.DataFrame(columns=combinations, index=['correlation'])

# 计算每个组合的相关系数
for combination in combinations:
    subset_df = df[list(combination)]
    corr_matrix = subset_df.corr()
    correlation = corr_matrix.stack().mean()
    result_df.loc['correlation', combination] = correlation

print(result_df)