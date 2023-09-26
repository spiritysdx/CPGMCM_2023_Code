import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
import numpy as np

# 示例数据
data7 = pd.read_excel('D:/DESKTOP/2b数据处理结果.xlsx')
data = data7[list(data7.columns)[13:]]

# 定义所需的聚类算法和参数
clustering_algorithms = [
    ("K-Means", KMeans())  # 不需要指定初始参数
]

# 定义KMeans的超参数搜索范围
param_grid = {
    'n_clusters': [3, 4, 5],  # 尝试不同的聚类数量
    'init': ['k-means++', 'random'],  # 不同的初始化方法
    'max_iter': [100, 200, 300],  # 不同的最大迭代次数
    'random_state': list(range(501))  # 0到500的随机数种子
}

# 创建GridSearchCV对象，启用多进程计算
grid_search = GridSearchCV(estimator=KMeans(), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)#多进程参数：, n_jobs=3)

# 执行网格搜索（此时会利用多进程）
grid_search.fit(data)

# 输出最佳参数和分数
print("Best Parameters:", grid_search.best_params_)
print("Best Negative Mean Squared Error:", grid_search.best_score_)

# 获取最佳参数的KMeans模型
best_kmeans_model = grid_search.best_estimator_

# 使用最佳模型进行聚类
best_kmeans_model.fit(data)
best_labels = best_kmeans_model.labels_

# 计算评估指标
silhouette = silhouette_score(data, best_labels)
ch_index = calinski_harabasz_score(data, best_labels)

print("Clustering Algorithm: K-Means (Tuned)")
print("Clustering Labels:", best_labels)
print("Silhouette coefficient:", silhouette)
print("Calinski-Harabasz index:", ch_index)