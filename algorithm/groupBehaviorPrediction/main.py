import datetime

import torch
import torch.optim as optim
from algorithm.groupBehaviorPrediction.DataLoader import Dataloader
# from model.ablation.LSTMPMAgbp import LSTMPMAGbp
# from model.ablation.GCNPMAgbp import GCNPMAGbp
# from model.ablation.LSTMGCNATTgbp import LSTMGCNATTGbp
# from model.ablation.LSTMGCNAVGgbp import LSTMGCNAVGGbp
from algorithm.groupBehaviorPrediction.LSTMGCNPMAgbp import LSTMGCNPMAGbp
import time
import math
import heapq
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def train(model, train_loader, lr, epoch, weight_decay):
    model.train()
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # loss_fun = torch.nn.MSELoss()
    loss_fun = torch.nn.BCELoss()
    eps = 1e-9
    total_loss = 0
    train_times = 0
    t1 = time.time()
    for batch_id, (group_inputs, act_inputs, act_seq_ixs, labels) in enumerate(train_loader):
        group_inputs, act_inputs, act_seq_ixs, labels = group_inputs.to(device), act_inputs.to(device), act_seq_ixs.to(device), labels.to(device)
        t2 = time.time()
        pred_y = model(group_inputs, act_inputs, act_seq_ixs)

        # loss = torch.mean((pred_y - labels)**2)
        # loss = loss_fun(pred_y, labels.float().view(-1,1))
        pred_y = torch.clamp(pred_y.view(-1),eps,1-eps)     # 防止ouput为0输入到log函数中
        # loss = torch.mean(-labels*torch.log(pred_y) - (1-labels)*torch.log(1-pred_y))
        loss = loss_fun(pred_y, labels.float())
        total_loss += loss.item()
        train_times += 1

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 500 == 0:
            print('Epoch:{} batch:{} time_cost:{:.2f}s loss:{:.6f}'.format(epoch, batch_id, time.time()-t2, loss.item()))

    print('|Epoch:{} end, training_time_cost:{:.2f}s total_loss:{:.6f}'.format(epoch, time.time()-t1, total_loss/train_times))



def evaluate(model, test_loader, K):
    model.eval()
    HRs = []
    NDCGs = []
    (group_testActs, group_testNagetives, test_act_seq_ix) = test_loader
    ix = 0
    total = len(group_testActs)
    for g_act, neg_acts, seq_ix in zip(group_testActs, group_testNagetives, test_act_seq_ix):
        g = g_act[0]
        pos_act = g_act[1]

        act_inputs = torch.LongTensor(neg_acts + [pos_act]).to(device)
        group_inputs = torch.LongTensor([g]*len(act_inputs)).to(device)
        act_seq_ixs = torch.LongTensor([seq_ix]*len(act_inputs)).to(device)
        pred_y = model(group_inputs, act_inputs, act_seq_ixs)

        act_scores = {}
        for act, score in zip(act_inputs, pred_y.view(-1)):
            act_scores[act.item()] = score.item()

        # 计算每种K值下的HR和NDCG
        hr_k = []
        ndcg_k = []
        for each_K in K:
            topk_acts = heapq.nlargest(each_K, act_scores, key=act_scores.get)
            hr = getHitRatio(topk_acts, pos_act)
            ndcg = getNDCG(topk_acts, pos_act)
            hr_k.append(hr)
            ndcg_k.append(ndcg)
        HRs.append(hr_k[:])
        NDCGs.append(ndcg_k[:])
        ix += 1
        n = (ix / total) * 10
        print('\rProgress：{:.2f}%'.format(n * 10), end='')
    HR_mean = torch.tensor(HRs, dtype=torch.float).mean(dim=0)
    NDCG_mean = torch.tensor(NDCGs, dtype=torch.float).mean(dim=0)

    return HR_mean.tolist(), NDCG_mean.tolist()

def getHitRatio(topk_acts, pos_act):
    if pos_act in topk_acts:
        return 1
    else:
        return 0

def getNDCG(topk_acts, pos_act):
    for i in range(len(topk_acts)):
        act = topk_acts[i]
        if act == pos_act:
            return math.log(2) / math.log(i+2)
    return 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def pretrain(model, user_train_loader, lr, epoch, tr_user_act_seqs, weight_decay):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fun = torch.nn.BCELoss()
    eps = 1e-9
    total_loss = 0
    train_times = 0
    t1 = time.time()
    for batch_id, (user_seq_ixs, user_targets, labels) in enumerate(user_train_loader):
        user_seq_ixs, user_targets, labels = user_seq_ixs.to(device), user_targets.to(device), labels.to(device)
        t2 = time.time()
        pred_y = model.user_forward(user_seq_ixs, user_targets, tr_user_act_seqs)

        pred_y = torch.clamp(pred_y.view(-1), eps, 1 - eps)  # 防止ouput为0输入到log函数中
        loss = loss_fun(pred_y, labels.float())
        total_loss += loss.item()
        train_times += 1

        # 更新模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 1000 == 0:
            print('Epoch:{} batch:{} time_cost:{:.2f}s loss:{:.6f}'.format(epoch, batch_id, time.time() - t2, loss.item()))

    print('|Epoch:{} end, training_time_cost:{:.2f}s total_loss:{:.6f}'.format(epoch, time.time() - t1, total_loss / train_times))

def user_evaluate(model, user_test_loader, K, te_user_act_seqs):
    model.eval()
    HRs = []
    NDCGs = []
    (te_user_seq_ixs, te_user_targets, user_testNagetives) = user_test_loader
    ix = 0
    total = len(te_user_seq_ixs)
    for seq_ix, user_target, neg_acts in zip(te_user_seq_ixs, te_user_targets, user_testNagetives):
        user_targets = torch.LongTensor(neg_acts + [user_target]).to(device)
        user_seq_ixs = torch.LongTensor([seq_ix] * len(user_targets)).to(device)
        pred_y = model.user_forward(user_seq_ixs, user_targets, te_user_act_seqs)

        act_scores = {}
        for act, score in zip(user_targets, pred_y.view(-1)):
            act_scores[act.item()] = score.item()

        topk_acts = heapq.nlargest(K, act_scores, key=act_scores.get)
        hr = getHitRatio(topk_acts, user_target)
        ndcg = getNDCG(topk_acts, user_target)
        HRs.append(hr)
        NDCGs.append(ndcg)

        ix += 1
        n = (ix / total) * 10
        print('\rProgress：{:.2f}%'.format(n * 10), end='')
    HR_mean = torch.tensor(HRs, dtype=torch.float).mean()
    NDCG_mean = torch.tensor(NDCGs, dtype=torch.float).mean()

    return HR_mean.item(), NDCG_mean.item()

def precision_evaluate(model, test_precision_data, K):
    model.eval()
    Precisions = []
    Recalls = []
    ix = 0
    total = len(test_precision_data)
    for g in test_precision_data.keys():
        pos_acts = test_precision_data[g]['pos_acts']
        neg_acts = test_precision_data[g]['neg_acts']
        seq_ixs = test_precision_data[g]['seq_ixs']

        act_inputs = torch.LongTensor(neg_acts + pos_acts).to(device)
        group_inputs = torch.LongTensor([g]*len(act_inputs)).to(device)
        act_seq_ixs = torch.LongTensor([seq_ixs[0]]*len(neg_acts) + seq_ixs).to(device)
        pred_y = model(group_inputs, act_inputs, act_seq_ixs)

        act_scores = {}
        for act, score in zip(act_inputs, pred_y.view(-1)):
            act_scores[act.item()] = score.item()

        # 计算每种K值下的HR和NDCG
        precision_k = []
        recall_k = []
        for each_K in K:
            topk_acts = heapq.nlargest(each_K, act_scores, key=act_scores.get)
            p, r = getPrecision_Recall(topk_acts, pos_acts)
            precision_k.append(p)
            recall_k.append(r)
        Precisions.append(precision_k[:])
        Recalls.append(recall_k[:])
        ix += 1
        n = (ix / total) * 10
        print('\rProgress：{:.2f}%'.format(n * 10), end='')
    Precision_mean = torch.tensor(Precisions, dtype=torch.float).mean(dim=0)
    Recall_mean = torch.tensor(Recalls, dtype=torch.float).mean(dim=0)
    return Precision_mean.tolist(), Recall_mean.tolist()

def getPrecision_Recall(topk_acts, pos_acts):
    tmp = [a for a in pos_acts if a in topk_acts]
    p = len(tmp) / len(topk_acts)
    r = len(tmp) / len(pos_acts)
    return p, r

# 加载用户预训练中的部分模型参数
def load_pretrain_model(model, path):
    print('加载预训练模型...')
    pretrain_dict = torch.load(path)
    model_dict = model.state_dict()
    n = 0
    for key in model_dict.keys():
        model_dict[key] = pretrain_dict[key]
        n += 1
        if n == 6:
            break
    model.load_state_dict(model_dict)

# if __name__ == '__main__':
#     starttime = datetime.datetime.now()
#     embed_size = 32
#     batch_size = 256
#     lr = 0.01
#     lr_user = 0.05
#     num_epoch = 10
#     train_rate = 0.8    # 训练集比例
#     num_negatives = 3
#     seq_length = 20
#     dropout = 0.2
#     weight_decay = 1e-6
#     K = [1, 3, 5, 10, 15]
#     seed = 42
#     pretrain_user = False
#
#
#     setup_seed(seed)    # 设置随机数种子
#
#     print('embed_size = {},batch_size = {},lr = {}, num_epoch = {}, seq_length = {}, num_negatives = {}, seed = {}, dropout = {}, weight_decay = {}'\
#           .format(embed_size,batch_size,lr,num_epoch,seq_length,num_negatives, seed, dropout, weight_decay))
#     dataloader = Dataloader('../../static/data/groupBehaviorPrediction/yelp_m10a3')   #douban_g15000
#     print('用户个数：', len(dataloader.users))
#     print('群体个数：', len(dataloader.groups))
#     print('act个数：', dataloader.act_num)
#     print('节点个数：', dataloader.graph.number_of_nodes())
#     train_loader, test_loader, all_act_seqs = dataloader.get_dataloader(batch_size, train_rate, num_negatives)
#     print('\n群体行为数据加载完毕！')
#
#     if torch.cuda.is_available():
#         print('使用GPU！')
#         (train_act_seqs, test_act_seqs) = all_act_seqs
#         for i in range(len(train_act_seqs)):
#             g_act_seqs_tr = train_act_seqs[i]
#             train_act_seqs[i] = [seq.to(device) for seq in g_act_seqs_tr]
#         for i in range(len(test_act_seqs)):
#             g_act_seqs_te = test_act_seqs[i]
#             test_act_seqs[i] = [seq.to(device) for seq in g_act_seqs_te]
#
#         for g in dataloader.group_members:
#             dataloader.group_members[g] = dataloader.group_members[g].to(device)
#
#     aggregators = ["mean", "max", "min", "std"]
#     scalers = ["identity", "amplification", "attenuation"]
#
#     # aggregators = ["mean"]
#     print(aggregators, scalers)
#     print('LSTMPMAGbp模型')
#
#     model = LSTMGCNPMAGbp(num_act=dataloader.act_num,
#                           num_user=len(dataloader.users),
#                           embed_size=embed_size,
#                           group_members=dataloader.group_members,
#                           all_act_seqs=all_act_seqs,
#                           graph=dataloader.graph,
#                           aggregators=aggregators,
#                           scalers=scalers,
#                           dropout=dropout).to(device)
#
#     # #训练
#     # if pretrain_user:
#     #     user_train_loader, user_test_loader, (tr_user_act_seqs, te_user_act_seqs) = dataloader.get_user_dataloader(batch_size, num_negatives, seq_length=20)
#     #     if torch.cuda.is_available():
#     #         tr_user_act_seqs = [seq.to(device) for seq in tr_user_act_seqs]
#     #         te_user_act_seqs = [seq.to(device) for seq in te_user_act_seqs]
#     #     print('\n用户行为数据加载完毕！')
#     #     best = 0
#     #     for epoch in range(16):
#     #         pretrain(model, user_train_loader, lr=lr_user, epoch=epoch, tr_user_act_seqs=tr_user_act_seqs,weight_decay=weight_decay)
#     #
#     #         if epoch % 5 == 0 and epoch < 15:
#     #             lr_user *= 0.5
#     #         t = time.time()
#     #         HR, NDCG = user_evaluate(model, user_test_loader, K[2], te_user_act_seqs)
#     #         print('User evaluation|Epoch:{} evaluate_time_cost:{:.2f}s HR@{}:{:.6f} NDCG@{}:{:.6f}'.format(epoch, time.time() - t, K[2], HR, K[2], NDCG))
#     #         if HR > best:
#     #             best = HR
#     #             torch.save(model.state_dict(), '../../static/data/groupBehaviorPrediction/pretrain_pma_yp.pt')
#
#     # 加载预训练模型
#     # load_pretrain_model(model, '../../static/data/groupBehaviorPrediction/pretrain_pma_yp.pt')
#     # model.load_state_dict(torch.load('pretrain_model/pretrain_pma_yp.pt'))
#
#     # best_HR = [0, 0, 0, 0, 0]
#     # best_NDCG = [0, 0, 0, 0, 0]
#     # for epoch in range(num_epoch+1):
#     #     if epoch > 0 and epoch % 10 == 0:
#     #         lr *= 0.7
#     #     train(model,train_loader,lr,epoch,weight_decay)
#     #
#     #     if epoch <= 9:
#     #         if epoch % 3 == 0:
#     #             t = time.time()
#     #             HR, NDCG = evaluate(model, test_loader, K)
#     #             print('Model evaluation|Epoch:{} evaluate_time_cost:{:.2f}s HR:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} NDCG:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},'
#     #                   .format(epoch, time.time() - t, HR[0], HR[1], HR[2], HR[3], HR[4], NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4]))
#     #             if HR[2] > best_HR[2]:
#     #                 # torch.save(model.state_dict(), 'pma_yp.pt')
#     #                 best_HR = HR[:]
#     #                 best_NDCG = NDCG[:]
#     #             print('best_HR:', best_HR, ' best_NDCG', best_NDCG)
#     #     elif epoch > 9:
#     #         if epoch % 5 == 0 or epoch == num_epoch-1:
#     #             t = time.time()
#     #             HR, NDCG = evaluate(model, test_loader, K)
#     #             print('Model evaluation|Epoch:{} evaluate_time_cost:{:.2f}s HR:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f} NDCG:{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},'
#     #                   .format(epoch, time.time() - t, HR[0], HR[1], HR[2], HR[3], HR[4], NDCG[0], NDCG[1], NDCG[2], NDCG[3], NDCG[4]))
#     #             if HR[2] > best_HR[2]:
#     #                 torch.save(model.state_dict(), '../../static/data/groupBehaviorPrediction/pma_yp.pt')
#     #                 best_HR = HR[:]
#     #                 best_NDCG = NDCG[:]
#     #             print('best_HR:', best_HR, ' best_NDCG', best_NDCG)
#     model.load_state_dict(torch.load('../../static/data/groupBehaviorPrediction/pma_yp.pt'))
#     best_HR, best_NDCG = evaluate(model, test_loader, K)
#     test_precision_data = dataloader.get_test_precision_data()
#     precision, recall = precision_evaluate(model, test_precision_data, K)
#     HR, NDCG = evaluate(model, test_loader, K)
#     print('\nbest_HR:', HR)
#     print('best_NDCG:', NDCG)
#     print('precision:', precision)
#     print('recall:', recall)
#     endtime = datetime.datetime.now()
#     print(endtime - starttime)
#     # print(test_loader[0])
#     # print("xxxxxxxxxxxx")
#     # print(test_loader[1][0])
#     # print("yyyyyyyyyyyy")
#     # print(test_loader[2])
#     # print(model.group_members)

