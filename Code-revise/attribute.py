import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor


def evaluate_density(embeddings):
    """
    评估嵌入的稠密性及其他属性
    :param embeddings: numpy数组，形状为(num_entities, embedding_dim)，表示TransE的嵌入向量
    :return: 各项嵌入评估指标
    """
    # 计算每个向量的L2范数
    norms = np.linalg.norm(embeddings, axis=1)
    avg_norm = np.mean(norms)

    # 计算范数的标准差
    std_norm = np.std(norms)

    # 计算嵌入向量的熵
    min_val, max_val = embeddings.min(), embeddings.max()
    normalized_embeddings = (embeddings - min_val) / (max_val - min_val + 1e-9)
    flat_embeddings = normalized_embeddings.flatten()
    histogram, bin_edges = np.histogram(flat_embeddings, bins=50, density=True)
    bin_widths = np.diff(bin_edges)
    embedding_entropy = entropy(histogram * bin_widths)  # 熵值

    # 计算嵌入向量之间的平均余弦相似性
    cos_sim_matrix = cosine_similarity(embeddings)
    avg_cos_sim = np.mean(cos_sim_matrix[np.triu_indices_from(cos_sim_matrix, k=1)])

    # 计算均匀性：嵌入向量各维度的均值和方差
    embedding_mean = np.mean(embeddings, axis=0)
    embedding_variance = np.var(embeddings, axis=0)
    avg_variance = np.mean(embedding_variance)

    # 中心化程度：向量均值的L2范数，反映嵌入向量是否围绕原点
    centroid_norm = np.linalg.norm(embedding_mean)

    # 返回各项评估指标
    return {
        "avg_norm": avg_norm,
        "std_norm": std_norm,
        "embedding_entropy": embedding_entropy,
        "avg_cos_sim": avg_cos_sim,
        "avg_variance": avg_variance,
        "centroid_norm": centroid_norm
    }


def detect_anomalies_cluster_structure(embeddings, eps=0.5, min_samples=5):
    """
    基于向量簇结构检测异常嵌入点。
    :param embeddings: numpy数组，形状为(num_entities, embedding_dim)
    :param eps: DBSCAN中的半径参数
    :param min_samples: DBSCAN中的最小样本数
    :return: 异常点的比例
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings)

    # 计算异常点比例
    num_anomalies = np.sum(labels == -1)
    anomaly_ratio = num_anomalies / len(labels)

    return anomaly_ratio


def detect_attack_in_embeddings(embeddings, n_neighbors=5):
    """
    检测知识图谱嵌入向量的翻转攻击，并返回多个指标。
    :param embeddings: numpy数组，形状为(num_entities, embedding_dim)
    :param n_neighbors: 邻居数量
    :return: 一个包含多个指标的字典
    """
    # 1. 计算邻居距离
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embeddings)
    distances, _ = nbrs.kneighbors(embeddings)

    # 2. 计算平均距离
    avg_distances = np.mean(distances[:, 1:], axis=1)
    mean_avg_distance = avg_distances.mean()

    # 3. 计算局部离群因子 (LOF)
    lof = LocalOutlierFactor(n_neighbors=n_neighbors)
    lof_scores = -lof.fit_predict(embeddings)  # LOF scores, -1 indicates outliers

    # 4. 计算距离分布的标准差
    std_distances = np.std(distances[:, 1:], axis=1)
    mean_std_distance = std_distances.mean()

    # 5. 计算离群点的比例
    outlier_ratio = np.mean(lof_scores == -1)

    # 6. 获取最大离群因子
    max_lof_score = lof.negative_outlier_factor_.min()  # More negative means more outlying

    return mean_avg_distance


def exponential_normalize(array):
    """
    使用指数形式对数组进行归一化.
    :param array: 输入的一维 numpy 数组
    :return: 归一化后的数组
    """
    # 计算数组的最小值和最大值
    min_val = np.min(array)
    max_val = np.max(array)

    # 避免最小值和最大值相等的情况
    if min_val == max_val:
        return np.ones_like(array) * np.exp(array[0])

    # 将数组缩放到 [0, 1] 区间
    scaled_array = (array - min_val) / (max_val - min_val)

    # 使用指数函数进行归一化
    exp_normalized_array = np.exp(scaled_array)

    # 再次缩放到 [0, 1] 区间
    final_normalized_array = (exp_normalized_array - np.min(exp_normalized_array)) / (
                np.max(exp_normalized_array) - np.min(exp_normalized_array))

    return final_normalized_array


