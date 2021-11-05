# 数据加载   采用随机负采样方法
import pandas as pd
import networkx as nx
import torch
import torch.utils.data as Data
from torch.utils.data import random_split
import numpy as np
import copy

class Dataloader:
    def __init__(self, base_path):
        self.base_path = base_path                      # 数据基路径
        self.users = []                                 # 用户列表
        self.act_num = 0                                # 行为个数
        self.user_acts = self.load_user_acts()          # 用户行为
        self.group_members = self.load_group_members()  # 群体成员,字典类型
        self.group_acts = self.load_group_acts()        # 群体行为
        self.groups = self.group_acts['group'].tolist() # 群体列表
        self.graph = self.load_graph()                  # 图数据
        self.num = 0
        self.num2 = 0

    def load_user_acts(self):
        user_acts = pd.read_csv(self.base_path+'/user_events.dat',sep='\t',header=None, names=['user','acts'])
        user_acts['acts'] = user_acts['acts'].map(self.str_to_list)
        self.users = user_acts['user'].drop_duplicates().tolist()
        # 获取acts集合
        all_acts = []
        for i, line in user_acts.iterrows():
            acts = line['acts']
            all_acts.extend(acts)
        # print('len(all_acts)',len(all_acts))
        self.act_num = len(set(all_acts))
        # print('行为数量：',self.act_num)
        # print('最大编号：',max(all_acts))
        return user_acts

    def load_group_members(self):
        group_members = pd.read_csv(self.base_path+'/groupid_members.dat',sep='\t',header=None,names=['group','members'])
        group_members['members'] = group_members['members'].map(self.str_to_list)
        g_m_d = {}      # 群体成员字典
        for i, line in group_members.iterrows():
            g = line['group']
            m = line['members']
            g_m_d.update({g : torch.LongTensor(m)})
        return g_m_d

    def load_group_acts(self):
        group_acts = pd.read_csv(self.base_path+'/groupid_events.dat',sep='\t',header=None,names=['group','acts'])
        group_acts['acts'] = group_acts['acts'].map(self.str_to_list)
        return group_acts

    def load_graph(self):
        pd_edges = pd.read_csv(self.base_path+'/graph.txt', sep=' ', header=None, names=['src','tar'])
        pd_edges['src'] = pd_edges['src'].map(int)
        pd_edges['tar'] = pd_edges['tar'].map(int)
        # 删除没有行为数据的用户
        pd_edges = pd_edges[pd_edges['src'].isin(self.users)]
        pd_edges = pd_edges[pd_edges['tar'].isin(self.users)]
        edges = pd_edges.values.tolist()
        graph = nx.Graph(edges)
        return graph

    # 将读取到的字符串转换成列表，同时将列表内的元素转换为整型
    def str_to_list(self, str_data):
        data = eval(str_data)
        new_data = [int(data[i]) for i in range(len(data))]
        return new_data

    #######################群体数据构建########################
    # 将group_acts转换为单条形式
    def get_pos_group_data(self):
        all_groups = []
        all_acts = []
        for i, line in self.group_acts.iterrows():
            group = line['group']
            acts = line['acts']
            for act in acts:
                all_groups.append(group)
                all_acts.append(act)
        return all_groups, all_acts

    # 构造dataset, 将群体行为数据划分为训练集和测试集
    def split_data(self, train_rate):
        all_groups, all_acts = self.get_pos_group_data()
        pos_group_dataset = Data.TensorDataset(torch.LongTensor(all_groups), torch.LongTensor(all_acts))
        data_num = len(pos_group_dataset)
        train_num = round(train_rate * data_num)
        test_num = round((1 - train_rate) * data_num)
        print('随机切分群体行为数据...')
        pos_group_train_data, pos_group_test_data = random_split(pos_group_dataset, [train_num, test_num])
        # train_data, test_data = random_split(dataset,[train_num,test_num],generator=torch.Generator().manual_seed(25))
        print('训练集数量', len(pos_group_train_data))
        print('测试集数量', len(pos_group_test_data))
        return pos_group_train_data, pos_group_test_data

    # 负采样
    def negative_sampling(self, pos_dataset, num_negatives):
        print('群体行为负采样，采样个数：', num_negatives)
        group_input = []    # 群体集合，group_act_num * 1
        act_input = []      # 群体行为集合，group_act_num * 1
        act_seq_ix = []     # 成员行为序列索引
        label = []
        total = len(pos_dataset)
        for ix in range(total):
            (group, pos_act) = pos_dataset[ix]
            group_input.append(group.item())
            act_input.append(pos_act.item())
            act_seq_ix.append(ix)
            label.append(1.0)
            g_acts = self.group_acts[self.group_acts['group'] == group.item()].iloc[0, 1]
            # 随机采样n个负样本
            for _ in range(num_negatives):
                neg_act = np.random.randint(self.act_num)
                while neg_act in g_acts:
                    neg_act = np.random.randint(self.act_num)
                group_input.append(group.item())
                act_input.append(neg_act)
                act_seq_ix.append(ix)
                label.append(0.0)
        return group_input, act_input, act_seq_ix, label

    def test_negative_sampling(self, pos_testdata, num_negatives):
        print('\n测试集负采样，采样个数：', num_negatives)
        group_testActs = []
        group_testNagetives = []
        act_seq_ix = []  # 成员行为序列索引
        total = len(pos_testdata)
        for ix in range(total):
            (group, pos_act) = pos_testdata[ix]
            g_acts = self.group_acts[self.group_acts['group'] == group.item()].iloc[0, 1]
            group_testActs.append([group.item(), pos_act.item()])
            # 随机采样n个负样本
            negatives = []
            for _ in range(num_negatives):
                neg_act = np.random.randint(self.act_num)
                while neg_act in g_acts:
                    neg_act = np.random.randint(self.act_num)
                negatives.append(neg_act)
            group_testNagetives.append(negatives[:])
            act_seq_ix.append(ix)
        return group_testActs, group_testNagetives, act_seq_ix

    def get_dataloader(self, batch_size, train_rate, num_negatives):
        pos_train_data, pos_test_data = self.split_data(train_rate=train_rate)
        self.pos_train_data, self.pos_test_data = pos_train_data, pos_test_data
        # 训练集
        group_input, act_input, all_act_seqs, label = self.negative_sampling(pos_train_data, num_negatives)
        train_data = Data.TensorDataset(torch.LongTensor(group_input),torch.LongTensor(act_input),torch.LongTensor(all_act_seqs),torch.LongTensor(label))
        train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_act_seqs = self.get_all_act_seqs(pos_train_data)
        # 测试集
        group_testActs, group_testNagetives, test_act_seq_ix = self.test_negative_sampling(pos_test_data, num_negatives=100)
        test_loader = (group_testActs, group_testNagetives, test_act_seq_ix)
        test_act_seqs = self.get_all_act_seqs(pos_test_data)
        return train_loader, test_loader, (train_act_seqs, test_act_seqs)

    # 获取群体成员的行为序列
    def get_member_act_seq(self, group, act, seq_length):
        members = self.group_members[group].tolist()
        act_seqs = []
        for m in members:
            m_acts = self.user_acts[self.user_acts['user']==m].iloc[0,1]
            ix = m_acts.index(act)
            start = 0
            if ix >= seq_length:
                start = ix - seq_length
            seq = m_acts[start:ix]
            act_seqs.append(torch.LongTensor(seq))
        return act_seqs

    def get_all_act_seqs(self, pos_dataset, seq_length=10):
        print('成员行为片段提取...')
        all_act_seqs = []
        total = len(pos_dataset)
        for i in range(total):
            (group, pos_act) = pos_dataset[i]
            act_seqs = self.get_member_act_seq(group=group.item(), act=pos_act.item(), seq_length=seq_length)
            all_act_seqs.append(copy.deepcopy(act_seqs))
            n = (i / total) * 10
            print('\r进度：{:.2f}%'.format(n * 10), end='')
        return all_act_seqs

    # 准确率测试数据
    def get_test_precision_data(self, num_negatives=100):
        print('\n测试集负采样，采样个数：', num_negatives)
        test_precision_data = {}
        total = len(self.pos_test_data)
        for ix in range(total):
            (group, pos_act) = self.pos_test_data[ix]
            if group.item() in test_precision_data:
                test_precision_data[group.item()]['pos_acts'].append(pos_act.item())
                test_precision_data[group.item()]['seq_ixs'].append(ix)
            else:
                g_acts = self.group_acts[self.group_acts['group'] == group.item()].iloc[0, 1]
                # 随机采样n个负样本
                negatives = []
                for _ in range(num_negatives):
                    neg_act = np.random.randint(self.act_num)
                    while neg_act in g_acts:
                        neg_act = np.random.randint(self.act_num)
                    negatives.append(neg_act)
                test_precision_data.update({group.item():{'pos_acts': [pos_act.item()],
                                                          'neg_acts': negatives[:],
                                                          'seq_ixs' : [ix]}})
        return test_precision_data


    ##################################用户行为数据构建##################################
    def get_user_data_instance(self, pos_dataset, seq_length):
        users = []
        user_act_seqs = []
        user_targets = []
        total = len(pos_dataset)
        for i in range(total):
            (group, act) = pos_dataset[i]
            members = self.group_members[group.item()].tolist()
            for m in members:
                m_acts = self.user_acts[self.user_acts['user'] == m].iloc[0, 1]
                ix = m_acts.index(act.item())
                start = 0
                if ix >= seq_length:
                    start = ix - seq_length
                seq = m_acts[start:ix]
                # seq = m_acts[0:ix]
                users.append(m)
                user_act_seqs.append(torch.LongTensor(seq))
                user_targets.append(act.item())
            n = (i / total) * 10
            print('\r获取用户数据，进度：{:.2f}%'.format(n * 10), end='')
        return users, user_act_seqs, user_targets

    def user_negative_sampling(self, users, user_targets, num_negatives):
        print('\n用户行为负采样,采样个数：', num_negatives)
        all_user_seq_ixs = []
        all_targets = []
        all_labels = []
        total = len(users)
        for ix in range(total):
            all_user_seq_ixs.append(ix)
            all_targets.append(user_targets[ix])
            all_labels.append(1.0)
            m_acts = self.user_acts[self.user_acts['user'] == users[ix]].iloc[0, 1]
            for _ in range(num_negatives):
                neg_act = np.random.randint(self.act_num)
                while neg_act in m_acts:
                    neg_act = np.random.randint(self.act_num)
                all_user_seq_ixs.append(ix)
                all_targets.append(neg_act)
                all_labels.append(0.0)
            n = (ix / total) * 10
            print('\r进度：{:.2f}%'.format(n * 10), end='')
        return all_user_seq_ixs, all_targets, all_labels

    def user_test_negative_sampling(self, users, num_negatives):
        print('\n测试集用户行为负采样,采样个数：', num_negatives)
        te_user_seq_ixs = []
        user_testNagetives = []
        total = len(users)
        for ix in range(total):
            m_acts = self.user_acts[self.user_acts['user'] == users[ix]].iloc[0, 1]
            neg_acts = []
            for _ in range(num_negatives):
                neg_act = np.random.randint(self.act_num)
                while neg_act in m_acts:
                    neg_act = np.random.randint(self.act_num)
                neg_acts.append(neg_act)
            te_user_seq_ixs.append(ix)
            user_testNagetives.append(neg_acts[:])
            n = (ix / total) * 10
            print('\r进度：{:.2f}%'.format(n * 10), end='')
        return te_user_seq_ixs, user_testNagetives

    def get_user_dataloader(self, batch_size, num_negatives, seq_length=50):
        users, tr_user_act_seqs, user_targets = self.get_user_data_instance(self.pos_train_data, seq_length)
        all_user_seq_ixs, all_targets, all_labels = self.user_negative_sampling(users, user_targets, num_negatives)
        user_train_data = Data.TensorDataset(torch.LongTensor(all_user_seq_ixs), torch.LongTensor(all_targets), torch.LongTensor(all_labels))
        user_train_loader = Data.DataLoader(user_train_data, batch_size=batch_size, shuffle=True)
        te_users, te_user_act_seqs, te_user_targets = self.get_user_data_instance(self.pos_test_data, seq_length)
        te_user_seq_ixs, user_testNagetives = self.user_test_negative_sampling(te_users, num_negatives=50)
        user_test_loader = (te_user_seq_ixs, te_user_targets, user_testNagetives)
        return user_train_loader, user_test_loader, (tr_user_act_seqs, te_user_act_seqs)



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
if __name__ == '__main__':
    # setup_seed(20)
    # dataloader = Dataloader('data/db100')
    # print('user_acts\n', dataloader.user_acts)
    # # print('group_members\n', dataloader.group_members)
    # print('group_acts\n', dataloader.group_acts)
    # print('用户个数：',len(dataloader.users))
    # print('群体个数：',len(dataloader.groups))
    # print('act个数：', dataloader.act_num)
    # print('节点个数：',dataloader.graph.number_of_nodes())
    # dataloader.get_test_precision_data(num_negatives=2,train_rate=0.8)
    # # #
    # train_loader, test_loader, (train_act_seqs, test_act_seqs) = dataloader.get_dataloader(batch_size=8, train_rate=0.8, num_negatives=2)
    # # pos_num = 0
    # # neg_num = 0
    # for batch_id, (group_input, act_input, act_seq_ix, label) in enumerate(train_loader):
    #     print('batch_id', batch_id)
    #     print(group_input)
    #     print(act_input)
    #     print(act_seq_ix)
    #     print(label, label.sum())
        # act_seqs = train_act_seqs[act_seq_ix]
        # print(act_seqs)
        # break
    #     p = label.sum()
    #     n = len(label) - p
    #     pos_num += p
    #     neg_num += n
    #     # print(label)
    # print('测试集： pos_num:{}, neg_num:{}'.format(pos_num,neg_num))

    # (group_testActs, group_testNagetives, test_act_seq_ix) = test_loader
    # print(len(group_testActs), len(group_testNagetives), len(test_act_seq_ix))
    # print(group_testActs[0])
    # print(group_testNagetives[0])
    # print(test_act_seq_ix[0])
    # for g_a, negs, seq_ix in zip(group_testActs, group_testNagetives, test_act_seq_ix):
    #     print(g_a)
    #     print(negs)
    #     print(seq_ix)
    #     break


    # import time
    # print(torch.cuda.is_available())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # model = torch.nn.Linear(32,32)
    # input = torch.randn((10000, 32))
    # model = torch.nn.Embedding(30000,32)
    # input = torch.randint(0,30000,size=(10000,))
    # t0 = time.time()
    # for i in range(100):
    #     output = model(input)
    # t1 = time.time()
    # model = model.to(device)
    # input = input.to(device)
    # t2 = time.time()
    # for i in range(100):
    #     output = model(input)
    # t3 = time.time()
    # print(input)
    # print(output)
    # print('cpu:', t1-t0)
    # print('gpu:', t2-t1)
    # print('gpu:', t3 - t2)
    # a = torch.tensor([1]).to(device)
    # b = torch.tensor([a] * 5).to(device)
    # print(b)
    # # c = torch.LongTensor([a] * 5, device=a.device)
    # member_embeds_act = torch.Tensor()
    # member_embeds_act = torch.cat((member_embeds_act, b))
    # print(member_embeds_act.dim())
    # print(member_embeds_act)
    # print(a,a.item())


    # act = torch.tensor([1]).to(device)
    # act_ = torch.tensor(1).to(device)
    # act_n = act_.expand(5)
    # print(act_, act_n)
    # member_num = 4
    # act_n = act.expand(member_num)
    # print(act, act_n)
    # t1 = time.time()
    # for i in range(10000):
    #     b = torch.LongTensor([act] * member_num).to(device)
    # print(b)
    # t2 = time.time()
    # for i in range(10000):
    #     act_n = act.expand(member_num)
    # t3 = time.time()
    # print(t2 - t1, t3 - t2)

    # a = torch.tensor([0,1])
    # b = torch.tensor([1,3,4])
    # ls = []
    # ls.append(a)
    # ls.append(b)
    # ls = [x.to(device) for x in ls]
    # print(ls)
    # path = 'C:/Users/Lenovo/Desktop/vec.dat'
    # vec = pd.read_csv('C:/Users/Lenovo/Desktop/vec.dat', sep=' ')
    # print(vec)
    # with open(path,'rb') as f:
    #     for line in f:
    #         print(line.decode('GBK'))

    aa = [1,2,3]
    b = [3,1, 5]
    tmp = [a for a in aa if a in b]
    # print(tmp)
    # print(len(tmp) / len(aa))
