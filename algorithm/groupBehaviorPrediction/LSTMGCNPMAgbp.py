'''
模型：基于LSTM和GCN的群体行为预测
GCN邻居聚合，单层
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from algorithm.groupBehaviorPrediction.pmaLayer import PmaLayer


class LSTMGCNPMAGbp(nn.Module):
    def __init__(self, num_act, num_user, embed_size, group_members, all_act_seqs, graph, aggregators, scalers, dropout=0.):
        super(LSTMGCNPMAGbp,self).__init__()
        self.num_act = num_act              # 行为个数
        self.num_user = num_user            # 用户个数
        self.embed_size = embed_size        # 嵌入维度
        self.group_members = group_members  # 群体成员字典
        (self.train_act_seqs, self.test_act_seqs) = all_act_seqs    # 所有的提取用户行为序列，以act_seq_ix索引
        self.graph = graph                  # 图数据
        self.avg_m = self.get_avg_group_size()  # 平均群体大小

        self.act_embeds = nn.Embedding(num_act, self.embed_size)      # 行为嵌入
        self.user_embeds = nn.Embedding(num_user, self.embed_size)  # 用户嵌入
        self.lstmLayer = LSTMLayer(self.embed_size, self.embed_size)         # lstm用户嵌入层
        self.gcnLayer = GCNLayer(self.embed_size)           # gcn用户嵌入层
        self.groupAggregationLayer =  PmaLayer(embed_size, self.avg_m, aggregators, scalers, num_layers=3, dropout=dropout) # 群体成员嵌入聚合层
        self.predictLayer = PredictLayer(3*self.embed_size, dropout=dropout)  # 预测层

    def forward(self, group_inputs, act_inputs, act_seq_ixs):
        device = group_inputs.device
        group_embeds = torch.Tensor().to(device)
        all_act_embeds = self.act_embeds(Variable(act_inputs))
        if self.training:
            all_act_seqs = self.train_act_seqs
        else:
            all_act_seqs = self.test_act_seqs
        for group, act, act_seq_ix in zip(group_inputs, act_inputs, act_seq_ixs):
            members = self.group_members[group.item()]
            member_num = torch.tensor(len(members), device=act.device)
            act_seqs = all_act_seqs[act_seq_ix.item()]
            member_embeds_act = torch.Tensor().to(device)      # 包含行为信息的成员嵌入
            member_embeds_gcn = torch.Tensor().to(device)      # 包含网络结构信息的成员嵌入
            # get member_embeds_act
            # act_embeds = self.act_embeds(Variable(act.expand(member_num)))
            for act_seq in act_seqs:
                m_act_embeds = self.act_embeds(Variable(act_seq))
                m_embed_act = self.lstmLayer(m_act_embeds.view(1, -1, self.embed_size))
                member_embeds_act = torch.cat((member_embeds_act, m_embed_act), dim=0)
            # get member_embeds_gcn
            for member in members:
                member_embed = self.user_embeds(Variable(member)).view(1,-1)
                nbrs = self.graph.neighbors(member.item())
                nbr_embeds = self.user_embeds(Variable(torch.LongTensor(list(nbrs)).to(act.device)))
                cat_nbr_embeds = torch.cat((member_embed, nbr_embeds))
                m_embed_gcn = self.gcnLayer(cat_nbr_embeds)
                member_embeds_gcn = torch.cat((member_embeds_gcn, m_embed_gcn), dim=0)
            # cat_member_embeds = torch.cat((member_embeds_act, member_embeds_gcn), dim=1)    # act和gcn两个信息拼接
            fus_member_embeds = member_embeds_act + member_embeds_gcn
            group_embed = self.groupAggregationLayer(fus_member_embeds, member_num)
            group_embeds = torch.cat((group_embeds, group_embed), dim=0)
        y = self.predictLayer(group_embeds, all_act_embeds)
        return y

    def user_forward(self, user_seq_ixs, user_targets, user_act_seqs):
        user_embeds = torch.Tensor().to(user_seq_ixs.device)
        for user_seq_ix in user_seq_ixs:
            user_seq = user_act_seqs[user_seq_ix]
            act_embeds = self.act_embeds(Variable(user_seq))
            u_embed = self.lstmLayer(act_embeds.view(1, -1, self.embed_size))
            user_embeds = torch.cat((user_embeds, u_embed), dim=0)
        act_embeds = self.act_embeds(user_targets)
        y = torch.mul(user_embeds, act_embeds).sum(dim=1)
        y = torch.sigmoid(y)
        return y


    def get_avg_group_size(self):
        s = 0
        for g in self.group_members:
            s += len(self.group_members[g])
        avg_m = s / len(self.group_members)
        return torch.tensor(avg_m)

class LSTMLayer(nn.Module):
    def __init__(self, act_ebd_size, user_ebd_size):
        super(LSTMLayer, self).__init__()
        self.userEmbedding = nn.LSTM(input_size=act_ebd_size, hidden_size=user_ebd_size, batch_first=True)

    def forward(self, act_embeds):
        output, h = self.userEmbedding(act_embeds)
        return output[:,-1,:]

class GroupAggregateLayer(nn.Module):
    '''
    基于注意力机制的成员嵌入聚合
    att(ut,aj) = H*ReLU(Pu*ut + Pa*aj + b)
    alph(ut,aj) = t--softmax(att(ut,aj))
    g = t--(alph * ut)
    '''
    def __init__(self, user_ebd_size, act_ebd_size, group_ebd_size):
        super(GroupAggregateLayer, self).__init__()
        self.userLayer = nn.Linear(user_ebd_size, 16)
        self.actLayer = nn.Linear(act_ebd_size, 16)
        self.Layer2 = nn.Linear(16, 1)
        # self.outputLayer = nn.Linear(user_ebd_size,group_ebd_size)

    def forward(self, member_embeds, act_embeds):
        m_embeds = self.userLayer(member_embeds)
        a_embeds = self.actLayer(act_embeds)
        att = self.Layer2(F.relu(m_embeds + a_embeds))
        weight = F.softmax(att.view(1, -1), dim=1)
        m_aggregation = torch.matmul(weight, member_embeds)
        # group_embed = self.outputLayer(m_aggregation)
        group_embed = m_aggregation
        return group_embed

class PredictLayer(nn.Module):
    def __init__(self, input_size, dropout=0.):
        super(PredictLayer, self).__init__()
        self.fcLayers =nn.Sequential(
            nn.Linear(input_size, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8,1)
        )

    def forward(self, group_embeds, act_embeds):
        element_embeds = torch.mul(group_embeds, act_embeds)
        concat_embeds = torch.cat((element_embeds, group_embeds, act_embeds),dim=1)
        y = torch.sigmoid(self.fcLayers(concat_embeds))
        return y

class GCNLayer(nn.Module):
    '''
    单层GCN
    '''
    def __init__(self, user_ebd_size):
        super(GCNLayer,self).__init__()
        # self.liner = nn.Linear(input_dim, user_ebd_size)
        self.outputLayer = nn.Linear(user_ebd_size, user_ebd_size)

    def forward(self, nbr_embeds):
        # nbr_embeds = F.relu(self.liner(nbr_embeds))
        nbr_aggre = nbr_embeds.mean(dim=0).view((1,-1))
        # cat_embed = torch.cat((user_embed, nbr_aggre),dim=1)
        new_embed = F.relu(self.outputLayer(nbr_aggre))
        return new_embed

