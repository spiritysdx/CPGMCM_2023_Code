import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
# 假设你有一个名为data的图数据，包含节点特征和边信息
data = pd.read_excel('D:/DESKTOP/2b数据处理结果.xlsx')  # 构建你的图数据


# 计算节点之间的相似度矩阵
features = data[list(data.columns)[13:]].values
similarity_matrix = np.dot(features, features.T)

# 谱聚类
sc = SpectralClustering(n_clusters=3, affinity='precomputed')
sc.fit(similarity_matrix)

# 输出聚类结果
labels = sc.labels_
for i, node_id in enumerate(data['ID']):
    print("Node", node_id, "Cluster:", labels[i])

silhouette = silhouette_score(data[list(data.columns)[1:]], labels)
ch_index = calinski_harabasz_score(data[list(data.columns)[1:]], labels)


print("Clustering Labels:", labels)
print("Silhouette coefficient:", silhouette)
print("Calinski-Harabasz index:", ch_index)
print("---------------------------------------")