# here put the import lib
import os
import random
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm



import sklearn.cluster as cluster

def cluster_items(embeddings, n_clusters=100):
    """对物品embedding进行K-means聚类, 返回每个物品的聚类标签"""
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels

def get_cluster_embeddings(args, item_embeddings):
    """获取聚类中心embedding"""
    n_clusters = args.n_clusters if hasattr(args, 'n_clusters') else 100
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(item_embeddings)
    return kmeans.cluster_centers_


def set_seed(seed):
    '''Fix all of random seed for reproducible training'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # only add when conv in your model


def get_n_params(model):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def get_n_params_(parameter_list):
    '''Get the number of parameters of model'''
    pp = 0
    for p in list(parameter_list):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def unzip_data(data, aug=True, aug_num=0):

    res = []
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)

    return res


def unzip_data_with_user(data, aug=True, aug_num=0):

    res = []
    users = []
    user_id = 1
    
    if aug:
        for user in tqdm(data):

            user_seq = data[user]
            seq_len = len(user_seq)

            for i in range(aug_num+2, seq_len+1):
                
                res.append(user_seq[:i])
                users.append(user_id)

            user_id += 1

    else:
        for user in tqdm(data):

            user_seq = data[user]
            res.append(user_seq)
            users.append(user_id)
            user_id += 1

    return res, users


def concat_data(data_list):

    res = []

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])

    else:

        raise ValueError

    return res


def concat_aug_data(data_list):

    res = []

    train = data_list[0]
    valid = data_list[1]

    for user in train:

        if len(valid[user]) == 0:
            res.append([train[user][0]])
        
        else:
            res.append(train[user]+valid[user])

    return res


def concat_data_with_user(data_list):

    res = []
    users = []
    user_id = 1

    if len(data_list) == 2:

        train = data_list[0]
        valid = data_list[1]

        for user in train:

            res.append(train[user]+valid[user])
            users.append(user_id)
            user_id += 1
    
    elif len(data_list) == 3:

        train = data_list[0]
        valid = data_list[1]
        test = data_list[2]

        for user in train:

            res.append(train[user]+valid[user]+test[user])
            users.append(user_id)
            user_id += 1

    else:

        raise ValueError

    return res, users


def filter_data(data, thershold=5):
    '''Filter out the sequence shorter than threshold'''
    res = []

    for user in data:

        if len(user) > thershold:
            res.append(user)
        else:
            continue
    
    return res



def random_neq(l, r, s=[]):    # 在l-r之间随机采样一个数，这个数不能在列表s中
    
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def metric_report(data_rank, topk=[10, 20]):
    """计算整体HR和NDCG，支持多topk（默认10、20）"""
    res = {}
    total = len(data_rank)  # 总样本数
    for k in topk:
        ndcg = 0.0
        hr = 0.0
        for rank in data_rank:
            if rank < k:  # 排名在topk内则计入
                ndcg += 1 / np.log2(rank + 2)  # NDCG公式（rank从0开始）
                hr += 1  # HR公式（命中则+1）
        # 归一化后存入结果
        res[f'NDCG@{k}'] = ndcg / total
        res[f'HR@{k}'] = hr / total
    return res
# def metric_report(data_rank, topk=10):

#     NDCG, HT = 0, 0
    
#     for rank in data_rank:

#         if rank < topk:
#             NDCG += 1 / np.log2(rank + 2)
#             HT += 1

#     return {'NDCG@10': NDCG / len(data_rank),
#             'HR@10': HT / len(data_rank)}



# def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

#     if args is not None:
#         ts_short = args.ts_short
#         ts_long = args.ts_long
#     else:
#         ts_short = 10
#         ts_long = 20

#     NDCG_s, HT_s = 0, 0
#     NDCG_m, HT_m = 0, 0
#     NDCG_l, HT_l = 0, 0
#     count_s = len(data_len[data_len<ts_short+aug_len])
#     count_l = len(data_len[data_len>=ts_long+aug_len])
#     count_m = len(data_len) - count_s - count_l

#     for i, rank in enumerate(data_rank):

#         if rank < topk:

#             if data_len[i] < ts_short+aug_len:
#                 NDCG_s += 1 / np.log2(rank + 2)
#                 HT_s += 1
#             elif data_len[i] < ts_long+aug_len:
#                 NDCG_m += 1 / np.log2(rank + 2)
#                 HT_m += 1
#             else:
#                 NDCG_l += 1 / np.log2(rank + 2)
#                 HT_l += 1

#     return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
#             'Short HR@10': HT_s / count_s if count_s!=0 else 0,
#             'Medium NDCG@10': NDCG_m / count_m if count_m!=0 else 0,
#             'Medium HR@10': HT_m / count_m if count_m!=0 else 0,
#             'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
#             'Long HR@10': HT_l / count_l if count_l!=0 else 0,}

def metric_len_report(data_rank, data_len, topk=[10, 20], aug_len=0, args=None):
    """按序列长度分组计算HR/NDCG（Short: <ts_user，Long: ≥ts_user）"""
    if args is not None:
        ts_user = args.ts_user  # 从配置读取长度阈值
    else:
        ts_user = 10  # 默认阈值：短序列<10，长序列≥10
    
    res = {}
    # 计算短/长序列的样本数（与topk无关）
    count_s = len(data_len[data_len < ts_user + aug_len])  # Short序列数量
    count_l = len(data_len[data_len >= ts_user + aug_len])  # Long序列数量
    
    # 对每个topk计算短/长序列的指标
    for k in topk:
        ndcg_s = 0.0  # Short序列NDCG
        hr_s = 0.0    # Short序列HR
        ndcg_l = 0.0  # Long序列NDCG
        hr_l = 0.0    # Long序列HR
        
        for i, rank in enumerate(data_rank):
            if rank < k:
                # 判断序列长度属于Short还是Long
                if data_len[i] < ts_user + aug_len:
                    ndcg_s += 1 / np.log2(rank + 2)
                    hr_s += 1
                else:
                    ndcg_l += 1 / np.log2(rank + 2)
                    hr_l += 1
        
        # 存入结果（避免除以0）
        res[f'Short NDCG@{k}'] = ndcg_s / count_s if count_s != 0 else 0.0
        res[f'Short HR@{k}'] = hr_s / count_s if count_s != 0 else 0.0
        res[f'Long NDCG@{k}'] = ndcg_l / count_l if count_l != 0 else 0.0
        res[f'Long HR@{k}'] = hr_l / count_l if count_l != 0 else 0.0
    return res

# def metric_len_report(data_rank, data_len, topk=10, aug_len=0, args=None):

#     if args is not None:
#         ts_user = args.ts_user
#     else:
#         ts_user = 10

#     NDCG_s, HT_s = 0, 0
#     NDCG_l, HT_l = 0, 0
#     count_s = len(data_len[data_len<ts_user+aug_len])
#     count_l = len(data_len[data_len>=ts_user+aug_len])

#     for i, rank in enumerate(data_rank):

#         if rank < topk:

#             if data_len[i] < ts_user+aug_len:
#                 NDCG_s += 1 / np.log2(rank + 2)
#                 HT_s += 1
#             else:
#                 NDCG_l += 1 / np.log2(rank + 2)
#                 HT_l += 1

#     return {'Short NDCG@10': NDCG_s / count_s if count_s!=0 else 0, # avoid division of 0
#             'Short HR@10': HT_s / count_s if count_s!=0 else 0,
#             'Long NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
#             'Long HR@10': HT_l / count_l if count_l!=0 else 0,}

def metric_pop_report(data_rank, pop_dict, target_items, topk=[10, 20], aug_pop=0, args=None):
    """按物品流行度分组计算HR/NDCG（Tail: <ts_tail，Popular: ≥ts_tail）"""
    if args is not None:
        ts_tail = args.ts_item  # 从配置读取流行度阈值
    else:
        ts_tail = 20  # 默认阈值：长尾物品<20，热门物品≥20
    
    res = {}
    # 计算目标物品的流行度及Tail/Popular样本数
    item_pop = pop_dict[target_items.astype("int64")]  # 每个目标物品的流行度
    count_s = len(item_pop[item_pop < ts_tail + aug_pop])  # Tail物品数量
    count_l = len(item_pop[item_pop >= ts_tail + aug_pop])  # Popular物品数量
    
    # 对每个topk计算Tail/Popular的指标
    for k in topk:
        ndcg_s = 0.0  # Tail物品NDCG
        hr_s = 0.0    # Tail物品HR
        ndcg_l = 0.0  # Popular物品NDCG
        hr_l = 0.0    # Popular物品HR
        
        for i, rank in enumerate(data_rank):
            if i == 0:  # 跳过padding索引（若有）
                continue
            if rank < k:
                # 判断物品属于Tail还是Popular
                if item_pop[i] < ts_tail + aug_pop:
                    ndcg_s += 1 / np.log2(rank + 2)
                    hr_s += 1
                else:
                    ndcg_l += 1 / np.log2(rank + 2)
                    hr_l += 1
        
        # 存入结果（避免除以0）
        res[f'Tail NDCG@{k}'] = ndcg_s / count_s if count_s != 0 else 0.0
        res[f'Tail HR@{k}'] = hr_s / count_s if count_s != 0 else 0.0
        res[f'Popular NDCG@{k}'] = ndcg_l / count_l if count_l != 0 else 0.0
        res[f'Popular HR@{k}'] = hr_l / count_l if count_l != 0 else 0.0
    return res

# def metric_pop_report(data_rank, pop_dict, target_items, topk=10, aug_pop=0, args=None):
#     """
#     Report the metrics according to target item's popularity
#     item_pop: the array of the target item's popularity
#     """
#     if args is not None:
#         ts_tail = args.ts_item
#     else:
#         ts_tail = 20

#     NDCG_s, HT_s = 0, 0
#     NDCG_l, HT_l = 0, 0
#     item_pop = pop_dict[target_items.astype("int64")]
#     count_s = len(item_pop[item_pop<ts_tail+aug_pop])
#     count_l = len(item_pop[item_pop>=ts_tail+aug_pop])

#     for i, rank in enumerate(data_rank):

#         if i == 0:  # skip the padding index
#             continue

#         if rank < topk:

#             if item_pop[i] < ts_tail+aug_pop:
#                 NDCG_s += 1 / np.log2(rank + 2)
#                 HT_s += 1
#             else:
#                 NDCG_l += 1 / np.log2(rank + 2)
#                 HT_l += 1

#     return {'Tail NDCG@10': NDCG_s / count_s if count_s!=0 else 0,
#             'Tail HR@10': HT_s / count_s if count_s!=0 else 0,
#             'Popular NDCG@10': NDCG_l / count_l if count_l!=0 else 0,
#             'Popular HR@10': HT_l / count_l if count_l!=0 else 0,}


def metric_len_5group(pred_rank, seq_len, thresholds=[5, 10, 15, 20], topk=[10, 20]):
    """按序列长度分5组（基于thresholds），计算每个topk的HR/NDCG"""
    hr_list = []  # 存储每个topk的5分组HR
    ndcg_list = []  # 存储每个topk的5分组NDCG
    total_samples = len(pred_rank)
    
    # 计算5个分组的样本数（与topk无关）
    count = np.zeros(5)
    count[0] = len(seq_len[seq_len < thresholds[0]])  # 组0：<5
    count[1] = len(seq_len[(seq_len >= thresholds[0]) & (seq_len < thresholds[1])])  # 组1：5~9
    count[2] = len(seq_len[(seq_len >= thresholds[1]) & (seq_len < thresholds[2])])  # 组2：10~14
    count[3] = len(seq_len[(seq_len >= thresholds[2]) & (seq_len < thresholds[3])])  # 组3：15~19
    count[4] = len(seq_len[seq_len >= thresholds[3]])  # 组4：≥20
    
    # 对每个topk计算5分组指标
    for k in topk:
        hr = np.zeros(5)
        ndcg = np.zeros(5)
        for i, rank in enumerate(pred_rank):
            target_len = seq_len[i]
            if rank < k:
                # 判断序列属于哪个分组
                if target_len < thresholds[0]:
                    ndcg[0] += 1 / np.log2(rank + 2)
                    hr[0] += 1
                elif target_len < thresholds[1]:
                    ndcg[1] += 1 / np.log2(rank + 2)
                    hr[1] += 1
                elif target_len < thresholds[2]:
                    ndcg[2] += 1 / np.log2(rank + 2)
                    hr[2] += 1
                elif target_len < thresholds[3]:
                    ndcg[3] += 1 / np.log2(rank + 2)
                    hr[3] += 1
                else:
                    ndcg[4] += 1 / np.log2(rank + 2)
                    hr[4] += 1
        
        # 归一化（避免除以0）
        for j in range(5):
            hr[j] = hr[j] / count[j] if count[j] != 0 else 0.0
            ndcg[j] = ndcg[j] / count[j] if count[j] != 0 else 0.0
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    
    return hr_list, ndcg_list, count

# def metric_len_5group(pred_rank, 
#                       seq_len, 
#                       thresholds=[5, 10, 15, 20], 
#                       topk=10):

#     NDCG = np.zeros(5)
#     HR = np.zeros(5)    
#     for i, rank in enumerate(pred_rank):

#         target_len = seq_len[i]
#         if rank < topk:

#             if target_len < thresholds[0]:
#                 NDCG[0] += 1 / np.log2(rank + 2)
#                 HR[0] += 1

#             elif target_len < thresholds[1]:
#                 NDCG[1] += 1 / np.log2(rank + 2)
#                 HR[1] += 1

#             elif target_len < thresholds[2]:
#                 NDCG[2] += 1 / np.log2(rank + 2)
#                 HR[2] += 1

#             elif target_len < thresholds[3]:
#                 NDCG[3] += 1 / np.log2(rank + 2)
#                 HR[3] += 1

#             else:
#                 NDCG[4] += 1 / np.log2(rank + 2)
#                 HR[4] += 1

#     count = np.zeros(5)
#     count[0] = len(seq_len[seq_len>=0]) - len(seq_len[seq_len>=thresholds[0]])
#     count[1] = len(seq_len[seq_len>=thresholds[0]]) - len(seq_len[seq_len>=thresholds[1]])
#     count[2] = len(seq_len[seq_len>=thresholds[1]]) - len(seq_len[seq_len>=thresholds[2]])
#     count[3] = len(seq_len[seq_len>=thresholds[2]]) - len(seq_len[seq_len>=thresholds[3]])
#     count[4] = len(seq_len[seq_len>=thresholds[3]])

#     for j in range(5):
#         NDCG[j] = NDCG[j] / count[j]
#         HR[j] = HR[j] / count[j]

#     return HR, NDCG, count

def metric_pop_5group(pred_rank, pop_dict, target_items, thresholds=[10, 30, 60, 100], topk=[10, 20]):
    """按物品流行度分5组（基于thresholds），计算每个topk的HR/NDCG"""
    hr_list = []  # 存储每个topk的5分组HR
    ndcg_list = []  # 存储每个topk的5分组NDCG
    item_pop = pop_dict[target_items.astype("int64")]  # 目标物品的流行度
    
    # 计算5个分组的样本数（与topk无关）
    count = np.zeros(5)
    count[0] = len(item_pop[item_pop < thresholds[0]])  # 组0：<10
    count[1] = len(item_pop[(item_pop >= thresholds[0]) & (item_pop < thresholds[1])])  # 组1：10~29
    count[2] = len(item_pop[(item_pop >= thresholds[1]) & (item_pop < thresholds[2])])  # 组2：30~59
    count[3] = len(item_pop[(item_pop >= thresholds[2]) & (item_pop < thresholds[3])])  # 组3：60~99
    count[4] = len(item_pop[item_pop >= thresholds[3]])  # 组4：≥100
    
    # 对每个topk计算5分组指标
    for k in topk:
        hr = np.zeros(5)
        ndcg = np.zeros(5)
        for i, rank in enumerate(pred_rank):
            target_pop = item_pop[i]
            if rank < k:
                # 判断物品属于哪个分组
                if target_pop < thresholds[0]:
                    ndcg[0] += 1 / np.log2(rank + 2)
                    hr[0] += 1
                elif target_pop < thresholds[1]:
                    ndcg[1] += 1 / np.log2(rank + 2)
                    hr[1] += 1
                elif target_pop < thresholds[2]:
                    ndcg[2] += 1 / np.log2(rank + 2)
                    hr[2] += 1
                elif target_pop < thresholds[3]:
                    ndcg[3] += 1 / np.log2(rank + 2)
                    hr[3] += 1
                else:
                    ndcg[4] += 1 / np.log2(rank + 2)
                    hr[4] += 1
        
        # 归一化（避免除以0）
        for j in range(5):
            hr[j] = hr[j] / count[j] if count[j] != 0 else 0.0
            ndcg[j] = ndcg[j] / count[j] if count[j] != 0 else 0.0
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
    
    return hr_list, ndcg_list, count

# def metric_pop_5group(pred_rank, 
#                       pop_dict, 
#                       target_items, 
#                       thresholds=[10, 30, 60, 100], 
#                       topk=10):

#     NDCG = np.zeros(5)
#     HR = np.zeros(5)    
#     for i, rank in enumerate(pred_rank):

#         target_pop = pop_dict[int(target_items[i])]
#         if rank < topk:

#             if target_pop < thresholds[0]:
#                 NDCG[0] += 1 / np.log2(rank + 2)
#                 HR[0] += 1

#             elif target_pop < thresholds[1]:
#                 NDCG[1] += 1 / np.log2(rank + 2)
#                 HR[1] += 1

#             elif target_pop < thresholds[2]:
#                 NDCG[2] += 1 / np.log2(rank + 2)
#                 HR[2] += 1

#             elif target_pop < thresholds[3]:
#                 NDCG[3] += 1 / np.log2(rank + 2)
#                 HR[3] += 1

#             else:
#                 NDCG[4] += 1 / np.log2(rank + 2)
#                 HR[4] += 1

#     count = np.zeros(5)
#     pop = pop_dict[target_items.astype("int64")]
#     count[0] = len(pop[pop>=0]) - len(pop[pop>=thresholds[0]])
#     count[1] = len(pop[pop>=thresholds[0]]) - len(pop[pop>=thresholds[1]])
#     count[2] = len(pop[pop>=thresholds[1]]) - len(pop[pop>=thresholds[2]])
#     count[3] = len(pop[pop>=thresholds[2]]) - len(pop[pop>=thresholds[3]])
#     count[4] = len(pop[pop>=thresholds[3]])

#     for j in range(5):
#         NDCG[j] = NDCG[j] / count[j]
#         HR[j] = HR[j] / count[j]

#     return HR, NDCG, count



def seq_acc(true, pred):

    true_num = np.sum((true==pred))
    total_num = true.shape[0] * true.shape[1]

    return {'acc': true_num / total_num}


def load_pretrained_model(pretrain_dir, model, logger, device):

    logger.info("Loading pretrained model ... ")
    checkpoint_path = os.path.join(pretrain_dir, 'pytorch_model.bin')

    model_dict = model.state_dict()

    # To be compatible with the new and old version of model saver
    try:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    except:
        pretrained_dict = torch.load(checkpoint_path, map_location=device)

    # filter out required parameters
    new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(new_dict)
    # 打印出来，更新了多少的参数
    logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
    model.load_state_dict(model_dict)

    return model


def record_csv(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)



def record_group(args, res_dict, path='log'):
    
    path = os.path.join(path, args.dataset)

    if not os.path.exists(path):
        os.makedirs(path)

    record_file = args.model_name + '.csv'
    csv_path = os.path.join(path, record_file)
    model_name = args.aug_file + '-' + args.now_str
    columns = list(res_dict.keys())
    columns.insert(0, "model_name")
    res_dict["model_name"] = model_name
    # columns = ["model_name", "HR@10", "NDCG@10", "Short HR@10", "Short NDCG@10", "Medium HR@10", "Medium NDCG@10", "Long HR@10", "Long NDCG@10",]
    new_res_dict = {key: [value] for key, value in res_dict.items()}
    
    if not os.path.exists(csv_path):

        df = pd.DataFrame(new_res_dict)
        df = df[columns]    # reindex the columns
        df.to_csv(csv_path, index=False)

    else:

        df = pd.read_csv(csv_path)
        add_df = pd.DataFrame(new_res_dict)
        df = pd.concat([df, add_df])
        df.to_csv(csv_path, index=False)
