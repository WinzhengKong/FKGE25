import pickle

# 加载最低相似度实体对字典
with open('lowest_similarity_pairs.pkl', 'rb') as f:
    lowest_similarity_dict = pickle.load(f)

# 读取原始的 train.txt 数据
with open('train.txt', 'r') as f:
    lines = f.readlines()

# 创建一个新的列表来保存修改后的数据
modified_lines = []

# 遍历所有数据
for line in lines:
    # 将每行数据按制表符分隔
    parts = line.strip().split('\t')

    # 遍历每个实体对（即每一列）
    for i in range(len(parts)):
        entity = parts[i]

        # 如果当前实体在最低相似度字典中，进行替换
        if entity in lowest_similarity_dict:
            # 获取该实体的相似度最低的实体
            lowest_entity = lowest_similarity_dict[entity]['lowest_entity']
            parts[i] = lowest_entity  # 替换当前实体

    # 将修改后的数据添加到新的列表中
    modified_lines.append('\t'.join(parts))

# 将修改后的数据保存到新的文件 'train.txt'
with open('train.txt', 'w') as f:
    f.writelines([line + '\n' for line in modified_lines])

print(f"所有数据已进行投毒处理，并保存到 'train.txt' 文件中。")
