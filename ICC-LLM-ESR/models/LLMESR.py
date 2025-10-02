# here put the import lib
import torch
import torch.nn as nn
from models.DualLLMSRS import DualLLMSASRec, DualLLMGRU4Rec, DualLLMBert4Rec
from models.utils import Contrastive_Loss2, ClusterHandler  # 导入精简后的ClusterHandler


class LLMESR_SASRec(DualLLMSASRec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 模糊约束强度

        # 初始化ClusterHandler（仅保留模糊逻辑）
        self.cluster_handler = ClusterHandler(
            dataset=args.dataset,
            hidden_size=args.hidden_size,
            device=device
        )

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"不支持的用户相似度函数: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions,** kwargs)  # 基础损失
        
        # 计算用户对齐损失
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]    # (bs*sim_num, hidden_size)
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)  # (bs, sim_num, hidden_size)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        # 根据相似度函数计算对齐损失
        if self.user_sim_func == "cl":
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        # 物品正则化损失（若启用）
        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        # 计算模糊-密度约束损失
        item_ids = seq[seq > 0]  # 过滤有效物品ID
        item_embeddings = self.id_item_emb(item_ids)  # 获取物品嵌入
        cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
        loss += self.gamma * cluster_loss

        # 整合所有损失
        loss += self.alpha * align_loss

        return loss


class LLMESR_GRU4Rec(DualLLMGRU4Rec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 模糊约束强度

        # 初始化ClusterHandler（仅保留模糊逻辑）
        self.cluster_handler = ClusterHandler(
            dataset=args.dataset,
            hidden_size=args.hidden_size,
            device=device
        )

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"不支持的用户相似度函数: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.beta = args.beta
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions,** kwargs)  # 基础损失
        
        # 计算用户对齐损失
        log_feats = self.log2feats(seq)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq)[:, -1, :]    
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        # 根据相似度函数计算对齐损失
        if self.user_sim_func == "cl":
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        # 物品正则化损失（若启用）
        if self.item_reg:
            unfold_item_id = torch.masked_select(seq, seq>0)
            llm_item_emb = self.adapter(self.llm_item_emb(unfold_item_id))
            id_item_emb = self.id_item_emb(unfold_item_id)
            reg_loss = self.reg(llm_item_emb, id_item_emb)
            loss += self.beta * reg_loss

        # 计算模糊-密度约束损失
        item_ids = seq[seq > 0]  # 过滤有效物品ID
        item_embeddings = self.id_item_emb(item_ids)  # 获取物品嵌入
        cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
        loss += self.gamma * cluster_loss

        # 整合所有损失
        loss += self.alpha * align_loss

        return loss


class LLMESR_Bert4Rec(DualLLMBert4Rec):

    def __init__(self, user_num, item_num, device, args):
        super().__init__(user_num, item_num, device, args)
        self.alpha = args.alpha
        self.user_sim_func = args.user_sim_func
        self.item_reg = args.item_reg
        self.gamma = args.gamma  # 模糊约束强度

        # 初始化ClusterHandler（仅保留模糊逻辑）
        self.cluster_handler = ClusterHandler(
            dataset=args.dataset,
            hidden_size=args.hidden_size,
            device=device
        )

        if self.user_sim_func == "cl":
            self.align = Contrastive_Loss2()
        elif self.user_sim_func == "kd":
            self.align = nn.MSELoss()
        else:
            raise ValueError(f"不支持的用户相似度函数: {self.user_sim_func}")

        self.projector1 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)
        self.projector2 = nn.Linear(2*args.hidden_size, 2*args.hidden_size)

        if self.item_reg:
            self.reg = Contrastive_Loss2()

        self._init_weights()


    def forward(self, 
                seq, 
                pos, 
                neg, 
                positions,
                **kwargs):
        
        loss = super().forward(seq, pos, neg, positions,** kwargs)  # 基础损失
        
        # 计算用户对齐损失
        log_feats = self.log2feats(seq, positions)[:, -1, :]
        sim_seq, sim_positions = kwargs["sim_seq"].view(-1, seq.shape[1]), kwargs["sim_positions"].view(-1, seq.shape[1])
        sim_num = kwargs["sim_seq"].shape[1]
        sim_log_feats = self.log2feats(sim_seq, sim_positions)[:, -1, :]
        sim_log_feats = sim_log_feats.detach().view(seq.shape[0], sim_num, -1)
        sim_log_feats = torch.mean(sim_log_feats, dim=1)

        # 根据相似度函数计算对齐损失
        if self.user_sim_func == "cl":
            align_loss = self.align(log_feats, sim_log_feats)
        elif self.user_sim_func == "kd":
            align_loss = self.align(log_feats, sim_log_feats)

        # 计算模糊-密度约束损失
        cluster_loss = self.calculate_cluster_loss(seq)

        # 整合所有损失
        loss += self.gamma * cluster_loss  # 模糊约束损失
        loss += self.alpha * align_loss    # 用户对齐损失

        return loss
        
    def calculate_cluster_loss(self, seq):
        """仅保留模糊损失计算逻辑，删除传统聚类相关过滤"""
        # 过滤有效物品ID（排除PAD和MASK）
        valid_mask = (seq > 0) & (seq != self.mask_token)        
        item_ids = seq[valid_mask]                             
        
        # 无有效物品时返回0损失
        if item_ids.numel() == 0:                                
            return torch.tensor(0.0, device=seq.device)
        
        # 直接计算模糊-密度约束损失
        item_embeddings = self.id_item_emb(item_ids)
        cluster_loss = self.cluster_handler.calculate_cluster_loss(item_ids, item_embeddings)
        return cluster_loss
