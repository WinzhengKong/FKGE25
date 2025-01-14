import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from pyod.models.ecod import ECOD
from sklearn.ensemble import IsolationForest
from attribute import *
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

def analyze_anomaly_scores(anomaly_scores_dict):

    for client_seq, scores in anomaly_scores_dict.items():
        scores = np.array(scores)  # 转换为 NumPy 数组以便计算
        print(f"Statistics for {client_seq}:")

        # 计算各类统计特征
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        median_score = np.median(scores)
        q25, q75 = np.percentile(scores, [25, 75])  # 计算四分位数

        # 输出统计特征
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Standard Deviation: {std_score:.4f}")
        print(f"  Min: {min_score:.4f}")
        print(f"  Max: {max_score:.4f}")
        print(f"  Median: {median_score:.4f}")
        print(f"  25th Percentile: {q25:.4f}")
        print(f"  75th Percentile: {q75:.4f}")
        print("-" * 40)


def trimmed_mean(users_grads, users_count, corrupted_count):
    number_to_consider = users_grads.shape[0] - corrupted_count - 1
    medians = np.median(users_grads, axis=0)
    differences = np.abs(users_grads - medians)
    indices = np.argsort(differences, axis=0)
    selected_indices = indices[:number_to_consider]
    selected_grads = np.take_along_axis(users_grads, selected_indices, axis=0)
    current_grads = np.mean(selected_grads, axis=0)
    return current_grads

def _krum_create_distances(local_global_embeddings):
    distances = {}
    for i, embed1 in enumerate(local_global_embeddings):
        distances[i] = {}
        for j, embed2 in enumerate(local_global_embeddings):
            if i != j:
                # 计算两个嵌入模型之间的欧氏距离
                distances[i][j] = np.linalg.norm(embed1 - embed2)
    return distances

def krum(local_global_embeddings, users_count, corrupted_count, distances=None, return_index=False):
    if not return_index:
        assert users_count >= 2 * corrupted_count + 1, (
        'users_count>=2*corrupted_count + 1', users_count, corrupted_count)

    non_malicious_count = users_count - corrupted_count
    minimal_error = 1e20
    minimal_error_index = -1

    # 如果没有预计算距离，计算所有用户之间的距离
    if distances is None:
        distances = _krum_create_distances(local_global_embeddings)

    # 遍历每个用户，计算其距离最近的 non_malicious_count 个用户的误差和
    for user in distances.keys():
        errors = sorted(distances[user].values())
        current_error = sum(errors[:non_malicious_count])  # 选择最近的 non_malicious_count 个用户
        if current_error < minimal_error:
            minimal_error = current_error
            minimal_error_index = user

    if return_index:
        return minimal_error_index
    else:
        return local_global_embeddings[minimal_error_index], minimal_error_index

def multi_krum(local_global_embeddings, users_count, corrupted_count, n, distances=None):
    assert users_count >= 2 * corrupted_count + 2 + n, (
        'users_count >= 2 * corrupted_count + 2 + n', users_count, corrupted_count, n)

    nb_in_score = users_count - corrupted_count - 2
    errors = []

    if distances is None:
        distances = _krum_create_distances(local_global_embeddings)

    for user in distances.keys():
        dists = sorted(distances[user].values())
        current_error = sum(dists[:nb_in_score])
        errors.append((current_error, user))

    # 按误差排序，选择前n个用户
    errors.sort()
    selected_indices = [user for _, user in errors[:n]]

    # 计算选中用户的平均梯度
    selected_grads = [local_global_embeddings[i] for i in selected_indices]
    mean_users_grads = np.mean(selected_grads, axis=0)

    return mean_users_grads, selected_indices

class Server:
    def __init__(self,args):
        self.args = deepcopy(args)
        self.nentity = self.load_entities()
    
    def load_entities(self):
        args = self.args
        local_file_dir = args.local_file_dir
        client_entities_dict = dict()
        for client_dir in os.listdir(local_file_dir):
            client_seq = int(client_dir)
            client_entities = []
            with open(os.path.join(local_file_dir,client_dir,"entities.dict"),"r",encoding="utf-8")as fin:
                for line in fin.readlines():
                    _,label = line.strip().split()
                    client_entities.append(label)
            client_entities_dict[client_seq] = client_entities
        self.client_entities_dict = client_entities_dict
        all_entities = []
        for client_seq in client_entities_dict.keys():
            all_entities.extend(client_entities_dict[client_seq])
        all_entities = list(set(all_entities))
        nentity = len(all_entities)
        client_entities_mapping = dict()
        for client_seq in client_entities_dict.keys():
            client_entities = client_entities_dict[client_seq]
            client_entities_mapping[client_seq] = [all_entities.index(client_entity) for client_entity in client_entities]
        self.client_entities_mapping = client_entities_mapping
        return nentity

    def generate_global_embedding(self):
        args = self.args
        hidden_dim = args.hidden_dim*2 if args.double_entity_embedding else args.hidden_dim
        embedding_range = torch.Tensor([(args.gamma+args.epsilon)/hidden_dim])
        self.global_entity_embedding = torch.zeros(self.nentity,hidden_dim)
        nn.init.uniform(tensor=self.global_entity_embedding,a=-embedding_range.item(),b=embedding_range.item())
    
    def assign_embedding(self):
        entity_embedding_dict = dict()
        for client_seq in self.client_entities_mapping.keys():
            client_embedding = self.global_entity_embedding[self.client_entities_mapping[client_seq]]
            entity_embedding_dict[client_seq] = client_embedding
        return entity_embedding_dict
    
    def aggregate_embedding(self,entity_embedding_dict,byzantine_client_indices):
        args = self.args
        hidden_dim = args.hidden_dim*2 if args.double_entity_embedding else args.hidden_dim

        '''
        embedding_distance = []
        print("Client evaluation:")
        for c_idx in range(0, args.client_num):
            distance = compute_density_with_dbscan(entity_embedding_dict[c_idx].cpu().detach().numpy())

            embedding_distance.append(distance)
        embedding_distance = np.array(embedding_distance)

        print(embedding_distance)
        '''

        later_global_embedding = torch.zeros(self.nentity,hidden_dim)
        weight = torch.zeros(self.nentity)
        # anomaly_scores_array = np.zeros(args.client_num)
        anomaly_scores_dict = {}
        if args.agg == "FedAD":
            # Load anomaly detector and use global train it (e.g. Isolation Forest)
            global_embedding_data = \
                self.global_entity_embedding. \
                    cpu().detach().numpy().reshape(-1, self.global_entity_embedding.shape[1])
            '''
            if args.component1 == 'normalization':
                print('Global normal!')
                global_embedding_data = global_embedding_data * np.mean(embedding_distance)
            '''
            if args.adm=="IF":

                isolation_forest = IsolationForest(n_estimators=200, contamination='auto', random_state=42)
                isolation_forest.fit(global_embedding_data)
                print('IF Model is trained!')
                for client_seq in entity_embedding_dict.keys():

                    client_embedding_data = entity_embedding_dict[client_seq].cpu().detach().numpy()

                    if len(client_embedding_data.shape) != 2:
                        raise ValueError(
                            f"Expected client embedding to be of shape (n, d), but got {client_embedding_data.shape}")
                    '''
                    if args.component1 == 'normalization':
                        print('Client normal!')
                        # client_embedding_data = scaler.fit_transform(client_embedding_data)
                        client_embedding_data = client_embedding_data * embedding_distance[client_seq]
                    '''
                    anomaly_scores = isolation_forest.decision_function(client_embedding_data)

                    anomaly_scores_dict[client_seq] = anomaly_scores.tolist()  # 转换为列表

                    client_average_scores = []
                    for client_seqq, scores in anomaly_scores_dict.items():
                        client_average = np.mean(scores)
                        client_average_scores.append(client_average)

            elif args.adm == "ECOD":
                # ECOD (TKDE 2022)
                clf = ECOD()
                clf.fit(global_embedding_data)
                print('ECOD Model is trained!')
                for client_seq in entity_embedding_dict.keys():

                    client_embedding_data = entity_embedding_dict[client_seq].cpu().detach().numpy()

                    if len(client_embedding_data.shape) != 2:
                        raise ValueError(
                            f"Expected client embedding to be of shape (n, d), but got {client_embedding_data.shape}")
                    '''
                    if args.component1 == 'normalization':
                    
                        print('normal!!')
                        # client_embedding_data = scaler.fit_transform(client_embedding_data)
                        client_embedding_data = client_embedding_data * embedding_distance[client_seq]
                    '''
                    anomaly_scores = clf.decision_function(client_embedding_data)

                    anomaly_scores_dict[client_seq] = anomaly_scores.tolist()

                    client_average_scores = []
                    for client_seqq, scores in anomaly_scores_dict.items():
                        client_average = np.mean(scores)
                        client_average_scores.append(client_average)

                # analyze_anomaly_scores(anomaly_scores_dict)
                client_average_scores = np.array(client_average_scores)
                print(client_average_scores)

                indices = np.argsort(-client_average_scores)

                processed_scores = np.ones_like(client_average_scores)

                if args.client_num == 5:
                    k = 2
                elif args.client_num == 10:
                    k = 4

                processed_scores[indices[:k]] = 0

                print(processed_scores)
                '''
                client_average_scores = np.array(client_average_scores)
                print(client_average_scores)
                result = client_average_scores
                # 计算全局平均值，并将大于平均值的赋值为1，小于等于平均值的赋值为0
                global_average_score = np.mean(result)
                print(global_average_score)
                processed_scores = np.where(result < global_average_score, 1, 0)
                '''
            '''
            # 转换为 NumPy 数组以便于计算
            local_global_embeddings_np = np.array(
                [embedding.numpy().flatten() for embedding in local_global_embeddings_list])
    
            global_entity_embedding_flat = self.global_entity_embedding.cpu().detach().numpy().flatten()
            '''

        if args.agg == "weighted" or args.agg == "FedAD":
            for client_seq in entity_embedding_dict.keys():
                if args.agg=="distance":
                    client_distance_norm = (self.global_entity_embedding[self.client_entities_mapping[client_seq]]-entity_embedding_dict[client_seq].cpu().detach()).norm(p=2,dim=1)
                    client_distance_norm = torch.exp(client_distance_norm)
                    weight[self.client_entities_mapping[client_seq]] += client_distance_norm
                    later_global_embedding[self.client_entities_mapping[client_seq]] += client_distance_norm.view(client_distance_norm.shape[0],1)*entity_embedding_dict[client_seq].cpu().detach()
                elif args.agg=="similarity":
                    client_similarity_score = torch.cosine_similarity(self.global_entity_embedding[self.client_entities_mapping[client_seq]],entity_embedding_dict[client_seq].cpu().detach(),dim=1)
                    client_similarity_score = torch.exp(client_similarity_score)
                    weight[self.client_entities_mapping[client_seq]] += client_similarity_score
                    later_global_embedding[self.client_entities_mapping[client_seq]] += client_similarity_score.view(client_similarity_score.shape[0],1)*entity_embedding_dict[client_seq].cpu().detach()
                elif args.agg=="weighted":
                    weight[self.client_entities_mapping[client_seq]] += 1
                    later_global_embedding[self.client_entities_mapping[client_seq]] += entity_embedding_dict[client_seq].cpu().detach()
                elif args.agg=="FedAD":
                    if processed_scores[client_seq] > 0:
                        weight[self.client_entities_mapping[client_seq]] += 1
                        later_global_embedding[self.client_entities_mapping[client_seq]] += \
                            entity_embedding_dict[client_seq].cpu().detach()
                        print(f'benign client embedding {client_seq} is added!')
                    else:
                        weight[self.client_entities_mapping[client_seq]] += 1
                        later_global_embedding[self.client_entities_mapping[client_seq]] += \
                            self.global_entity_embedding[self.client_entities_mapping[client_seq]].cpu().detach()
                        print(f'byzantine client embedding {client_seq} is processed!')
                elif args.agg=="FedSD":
                    if client_seq in byzantine_client_indices:
                        weight[self.client_entities_mapping[client_seq]] += 1
                        later_global_embedding[self.client_entities_mapping[client_seq]] += \
                            self.global_entity_embedding[self.client_entities_mapping[client_seq]].cpu().detach()
                        print(f'byzantine client embedding {client_seq} is processed!')
                    else:
                        weight[self.client_entities_mapping[client_seq]] += 1
                        later_global_embedding[self.client_entities_mapping[client_seq]] += \
                            entity_embedding_dict[client_seq].cpu().detach()
                        print(f'benign client embedding {client_seq} is added!')
                else:
                    raise ValueError("Aggregation method should be chosen among weighted/distance/similarity/FedAD")
        '''
        if args.agg == "Krum":
            print('Agg is Krum!')
            local_global_embeddings_list = []
            for client_seq in entity_embedding_dict.keys():
                local_global_embedding = torch.zeros_like(self.global_entity_embedding)
                for local_index, global_index in enumerate(self.client_entities_mapping[client_seq]):
                    if global_index < local_global_embedding.size(0):
                        local_global_embedding[global_index] = entity_embedding_dict[client_seq][
                            local_index].cpu().detach()

                local_global_embeddings_list.append(local_global_embedding)
            local_global_embeddings_list = np.array(local_global_embeddings_list)

            f = args.client_num // 2  # worse case 50% malicious points
            k = args.client_num - f - 1

            later_global_embedding, krum_idx = krum(local_global_embeddings_list, args.client_num, f - k)
            print(f"Krum select client {krum_idx}.")

        elif args.agg == "MultiKrum":
            print('Agg is MultiKrum!')
            local_global_embeddings_list = []
            for client_seq in entity_embedding_dict.keys():
                local_global_embedding = torch.zeros_like(self.global_entity_embedding)
                for local_index, global_index in enumerate(self.client_entities_mapping[client_seq]):
                    if global_index < local_global_embedding.size(0):
                        local_global_embedding[global_index] = entity_embedding_dict[client_seq][
                            local_index].cpu().detach()

                local_global_embeddings_list.append(local_global_embedding)
            local_global_embeddings_list = np.array(local_global_embeddings_list)

            f = args.client_num // 2  # worse case 50% malicious points
            k = args.client_num - f - 1

            later_global_embedding, Mutikrum_idx = multi_krum(local_global_embeddings_list
                                                          , args.client_num, f - k, f)
            print(f"Muti_Krum select client {Mutikrum_idx}.")

        elif args.agg == "TrimMedian":
            print('Agg is TrimMedian!')
            local_global_embeddings_list = []
            for client_seq in entity_embedding_dict.keys():
                local_global_embedding = torch.zeros_like(self.global_entity_embedding)
                for local_index, global_index in enumerate(self.client_entities_mapping[client_seq]):
                    if global_index < local_global_embedding.size(0):
                        local_global_embedding[global_index] = entity_embedding_dict[client_seq][
                            local_index].cpu().detach()

                local_global_embeddings_list.append(local_global_embedding)
            local_global_embeddings_list = np.array(local_global_embeddings_list)

            f = args.client_num // 2  # worse case 50% malicious points
            k = args.client_num - f - 1

            later_global_embedding = trimmed_mean(local_global_embeddings_list, args.client_num, k)
            print('~~~~debug~~~~')
        '''
        standard = torch.ones(weight.shape)
        weight = torch.where(weight>0,weight,standard)
        weight = weight.view(weight.shape[0],1)
        later_global_embedding/=weight
        self.global_entity_embedding = later_global_embedding
        #print("Global evaluation:")
        #global_distance = compute_density_with_dbscan(later_global_embedding.cpu().detach().numpy())

        #print(global_distance)