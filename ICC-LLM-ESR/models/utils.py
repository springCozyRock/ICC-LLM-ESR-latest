import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path 
import pickle
# 在 utils.py 顶部的 import 区添加
import math  # 新增：用于 sqrt 函数
# 精简版：仅支持模糊-密度约束损失
class ClusterHandler(nn.Module):
    def __init__(self, dataset, hidden_size, device, use_fuzzy=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.device = device
        self.dataset = dataset
        self.use_fuzzy = use_fuzzy  # 固定为True（因已删除传统逻辑，也可直接去掉该参数）

        # 仅保留模糊约束的初始化（必加载模糊文件）
        self._load_fuzzy_constraint_files()
        self.fuzzy_m = 1.8  # 模糊指数

    def _load_fuzzy_constraint_files(self):
        """加载模糊约束核心文件（hdbscan_fuzzy_U.pkl、hdbscan_cluster_centers.pkl）"""
        data_dir = Path(f'data/{self.dataset}/handled/')
        # 加载模糊隶属度向量
        fuzzy_U_path = data_dir / "hdbscan_fuzzy_U.pkl"
        if not fuzzy_U_path.exists():
            raise FileNotFoundError(f"模糊隶属度文件不存在：{fuzzy_U_path}")
        with open(fuzzy_U_path, 'rb') as f:
            self.fuzzy_U = torch.tensor(pickle.load(f), dtype=torch.float32, device=self.device)
        
        # 加载加权簇中心
        cluster_centers_path = data_dir / "hdbscan_cluster_centers.pkl"
        if not cluster_centers_path.exists():
            raise FileNotFoundError(f"加权簇中心文件不存在：{cluster_centers_path}")
        with open(cluster_centers_path, 'rb') as f:
            self.hdbscan_cluster_centers = torch.tensor(pickle.load(f), dtype=torch.float32, device=self.device)
        self.num_fuzzy_clusters = self.hdbscan_cluster_centers.shape[0]
        print(f"[Fuzzy Init] 成功加载模糊文件：物品数={self.fuzzy_U.shape[0]}, 簇数={self.num_fuzzy_clusters}")

    def calculate_cluster_loss(self, item_ids, item_embeddings):
        """仅计算模糊-密度约束损失"""
        if item_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        #print(f"[Fuzzy Loss] 处理物品数={item_ids.numel()} | 簇数={self.num_fuzzy_clusters}")
        return self._calculate_fuzzy_loss(item_ids, item_embeddings)

    def _calculate_fuzzy_loss(self, item_ids, item_embeddings):
        """模糊损失核心计算（动态原型+MSE）"""
        # 过滤无效物品ID（避免超出模糊隶属度向量范围）
        valid_mask = item_ids < self.fuzzy_U.shape[0]
        valid_ids = item_ids[valid_mask]
        valid_emb = item_embeddings[valid_mask]
        
        if valid_ids.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        
        # 计算动态模糊原型
        batch_fuzzy_U = self.fuzzy_U[valid_ids]
        fuzzy_weights = batch_fuzzy_U ** self.fuzzy_m
        prototypes = torch.matmul(
            fuzzy_weights.unsqueeze(1),
            self.hdbscan_cluster_centers.unsqueeze(0)
        ).squeeze(1)
        
        # MSE损失
        fuzzy_loss = F.mse_loss(valid_emb, prototypes)
        #print(f"[Fuzzy Loss] 当前批次损失值={fuzzy_loss.item():.6f}")
        return fuzzy_loss
# ---------------------- 以下为原有类（无需修改） ----------------------
class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)  # 恢复维度：(N, Length, C)
        outputs += inputs
        return outputs


class Contrastive_Loss2(nn.Module):
    def __init__(self, tau=1) -> None:
        super().__init__()
        self.temperature = tau

    def forward(self, X, Y):
        logits = (X @ Y.T) / self.temperature
        X_similarity = Y @ Y.T
        Y_similarity = X @ X.T
        targets = F.softmax((X_similarity + Y_similarity) / 2 * self.temperature, dim=-1)
        X_loss = self.cross_entropy(logits, targets, reduction='none')
        Y_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss = (Y_loss + X_loss) / 2.0
        return loss.mean()

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / math.sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention, V)
        return attention


class Multi_CrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num
        assert all_head_size % head_num == 0

        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)
        self.norm = math.sqrt(all_head_size)

    def forward(self, x, y, log_seqs):
        batch_size = x.size(0)
        # Q/K/V拆分多头
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1, 2)

        # 生成掩码（过滤PAD=0）
        attention_mask = (log_seqs == 0).unsqueeze(1).repeat(1, log_seqs.size(1), 1).unsqueeze(1)
        # 计算交叉注意力
        attention = CalculateAttention()(q_s, k_s, v_s, attention_mask)
        # 多头合并与输出
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        output = self.linear_output(attention)
        return output