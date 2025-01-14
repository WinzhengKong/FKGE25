import numpy as np
from scipy.stats import entropy
from sklearn.metrics.pairwise import cosine_similarity

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
