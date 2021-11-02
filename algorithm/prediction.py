import json
import pandas as pd
import networkx as nx
import numpy as np
import os
import sys
import random
import datetime
import math
import heapq

# k = 1# k 跳邻居
# #群体最大有301个个体，需要构造301个颜色list，总共39个颜色，循环使用
# colorList = ['#4f19c7', '#c71969', '#c71919', '#1984c7', '#c76919', '#8419c7', '#c79f19', '#c78419', '#c719b9',
#              '#199fc7', '#9f19c7', '#69c719', '#19c719', '#1919c7', '#c74f19', '#19c7b9', '#9fc719', '#c7b919',
#              '#b9c719', '#3419c7', '#19b9c7', '#34c719', '#19c784', '#c7199f', '#1969c7', '#c71984', '#1934c7',
#              '#84c719', '#194fc7', '#c7194f', '#19c74f', '#b919c7', '#19c769', '#19c79f', '#4fc719', '#c73419',
#              '#19c734', '#6919c7', '#c71934']
#画图网络面板最大尺寸，以及结点最大size
# -516.55884        min_x
# 712.18353         max_x
# -493.57944        min_y
# 522.4031          max_y
# 0                 min_size
# 66.66666666666667 max_size


#通过结点id查找结点颜色，-1表示未找到该id结点
import torch


def getColorById(id, nodes):
    for node in nodes:
        if node['id'] == id:
            return node['color']
    return -1

#从字典中获取所有都是该value值的key
def getDictKeys(dict, value):
    return [k for k,v in dict.items() if v == value]

#随机取一个群体（取一行，并且不在groupList内），总共有10437行(返回结果，groupid, members),2332
def getAGroup(groupList, test_group_loader, group_members):
    while True:
        groupNum=random.randint(0, len(test_group_loader[0]))
        #取第groupNum个
        groupid = test_group_loader[0][groupNum][0]
        #查找一个不在groupList内的群体，返回，否则再找
        if groupid not in groupList:
            # 获取该群体对应的成员列表
            temp_members = group_members[groupid]
            # 这里字符串要去掉空格
            members = [int(tensor.cpu().numpy()) for tensor in temp_members]
            # [groupid, pos_act]格式
            pos_act = test_group_loader[0][groupNum][1]
            neg_acts = test_group_loader[1][groupNum]
            act_seq_index = test_group_loader[2][groupNum]
            return groupid, members, pos_act, neg_acts, act_seq_index
    # while True:
    #     filename = os.path.dirname(os.path.dirname(__file__)) + '/static/data/groupBehaviorPrediction/yelp_m10a3/groupid_members.dat'
    #     groupNum=random.randint(0,10436)
    #     #跳过前groupNum行，往下读1行
    #     group = pd.read_csv(filename, sep='\t', header=None, names=['group', 'members'], skiprows=groupNum, nrows=1)
    #     group_records = group.to_dict(orient='records')
    #     groupid = group_records[0]['group']
    #     #将字符串[1,2,3, 4, 5,15]，去除前后端[]，用','分隔开，并去除前后端空格，转换为列表
    #     temp_members = group_records[0]['members'][1:-1].split(',')
    #     #这里字符串要去掉空格
    #     members = [s.strip() for s in temp_members]
    #     #查找一个不在groupList内的群体，返回，否则再找
    #     if groupid not in groupList:
    #         return groupid, members

#面板1000x1000，分8个象限（群体类别），根据不同的类别（不同象限），随机生成其对应区域的坐标
def getLocationByCategory(category):
    x = 0
    y = 0
    categories = '群体'
    if category == 0:
        y = random.randint(0, 1000)
        x = random.randint(0, y)
        categories = categories + 'A'
    elif category == 1:
        x = random.randint(0, 1000)
        y = random.randint(0, x)
        categories = categories + 'B'
    elif category == 2:
        x = random.randint(0, 1000)
        y = random.randint(-x, 0)
        categories = categories + 'C'
    elif category == 3:
        y = random.randint(-1000, 0)
        x = random.randint(0, -y)
        categories = categories + 'D'
    elif category == 4:
        y = random.randint(-1000, 0)
        x = random.randint(y, 0)
        categories = categories + 'E'
    elif category == 5:
        x = random.randint(-1000, 0)
        y = random.randint(x, 0)
        categories = categories + 'F'
    elif category == 6:
        x = random.randint(-1000, 0)
        y = random.randint(0, -x)
        categories = categories + 'G'
    elif category == 7:
        y = random.randint(0, 1000)
        x = random.randint(-y, 0)
        categories = categories + 'H'
    elif category == 8:
        y = random.randint(-1000, 1000)
        x = random.randint(-1000, 1000)
        categories = '邻居' + 'I'
    return x, y, categories

#面板1000x1000，分n x m个表格（群体类别），根据不同的类别（不同cell），随机生成其对应区域的坐标
def getLocationByCategory2(category, categoriesNum):
    lineNum = math.ceil(categoriesNum ** 0.5)
    #min_x =
    x = 0
    y = 0
    categories = '群体'
    if category == 0:
        y = random.randint(0, 1000)
        x = random.randint(0, y)
        categories = categories + 'A'
    elif category == 1:
        x = random.randint(0, 1000)
        y = random.randint(0, x)
        categories = categories + 'B'
    elif category == 2:
        x = random.randint(0, 1000)
        y = random.randint(-x, 0)
        categories = categories + 'C'
    elif category == 3:
        y = random.randint(-1000, 0)
        x = random.randint(0, -y)
        categories = categories + 'D'
    elif category == 4:
        y = random.randint(-1000, 0)
        x = random.randint(y, 0)
        categories = categories + 'E'
    elif category == 5:
        x = random.randint(-1000, 0)
        y = random.randint(x, 0)
        categories = categories + 'F'
    elif category == 6:
        x = random.randint(-1000, 0)
        y = random.randint(0, -x)
        categories = categories + 'G'
    elif category == 7:
        y = random.randint(0, 1000)
        x = random.randint(-y, 0)
        categories = categories + 'H'
    elif category == 8:
        y = random.randint(-1000, 1000)
        x = random.randint(-1000, 1000)
        categories = '邻居' + 'I'
    return x, y, categories

#面板1000x1000，随机生成坐标
def getLocationByCategory3(category):
    categories = '群体' + str(category)
    y = random.randint(-1000, 1000)
    x = random.randint(-1000, 1000)
    return x, y, categories

#一个群体的用户，前端展示信息，整理生成(方法参数,群体id，群体成员list，群体类别[0,1,2,3,4,5,6,7,8])
def groupDisplayInfo(userActList, maxsize, size_rate, groupid, members, category):
    nodes = []#前端展示需要封装的列表格式（结点信息）
    # 封装的结点集合，以及群体主体成员集合
    mainNodeSet = []#群体主体成员的集合
    # 将群体的主体结点载入结点集合（size授予最大）
    index = 0
    for member in members:
        mainNodeSet.append(member)
        x, y, categoryName = getLocationByCategory3(category)
        node = {}
        node['id'] = str(member)
        node['name'] = str(member)
        node['symbolSize'] = (1 + getActNumByNodeId(userActList, int(member)) * size_rate) % maxsize#应该改成这个人参加活动的多少增大结点
        node['x'] = x
        node['y'] = y
        node['value'] = int(groupid)
        node['category'] = category
        nodes.append(node)
        index = index + 1
    return mainNodeSet, nodes

#从当前封装的结点信息列表里面获取结点id列表
def getNodeIdListFromNodesList(nodes):
    nodeIdList = []
    for node in nodes:
        nodeIdList.append(int(node['id']))
    return nodeIdList

#太慢了，失败
#根据结点id找邻居结点所属的群体编号（并且不在现有展示的几个群体内），返回int型编号，参数nodeId为字符串
def getGroupIdByNoedId(existGroupList, nodeId):
    groupId = random.randint(-999999, 0)#未找到对应群体编号的，防止重叠
    filename = os.path.dirname(os.path.dirname(__file__)) + '/static/data/groupBehaviorPrediction/yelp_m10a3/groupid_members.dat'
    group = pd.read_csv(filename, sep='\t', header=None, names=['group', 'members'])
    group_records = group.to_dict(orient='records')
    for tempGroup in group_records:
        tempGroupId = tempGroup['group']
        if tempGroupId not in existGroupList:
            # 将字符串[1,2,3, 4, 5,15]，去除前后端[]，用','分隔开，并去除前后端空格，转换为列表
            temp_members = tempGroup['members'][1:-1].split(',')
            # 这里字符串要去掉空格
            tempMembers = [s.strip() for s in temp_members]
            if nodeId in tempMembers:
                return tempGroupId
    return groupId

#根据结点id找邻居结点所属的群体编号（并且不在现有展示的几个群体内），返回int型编号，参数nodeId为字符串
def getGroupIdByNoedId2(existGroupList, nodeId, user_groups):
    groupId = random.randint(-999999, 0)#未找到对应群体编号的，防止重叠
    #以下判断条件，过滤未加入任何群体的，人
    if nodeId.strip() in user_groups.keys():
        participateGroupIdList = user_groups[nodeId.strip()]
        for tempGroupId in participateGroupIdList:
            if tempGroupId not in existGroupList:
                return tempGroupId
    return groupId

#根据用户编号，获取用户参与活动集合，输入整数
def getActsByNodeId(userActList, nodeId):
    actListStr = userActList[nodeId][1]
    # 将字符串[1,2,3, 4, 5,15]，去除前后端[]，用','分隔开，并去除前后端空格，转换为列表
    temp_actList = actListStr[1:-1].split(',')
    # 这里字符串要去掉空格
    actList = [s.strip() for s in temp_actList]
    return actList

#根据用户编号，获取用户参与活动数量，输入整数
def getActNumByNodeId(userActList, nodeId):
    actListStr = userActList[nodeId][1]
    # 将字符串[1,2,3, 4, 5,15]，去除前后端[]，用','分隔开，并去除前后端空格，转换为列表
    temp_actList = actListStr[1:-1].split(',')
    return len(temp_actList)

# 彩蝶程序里面的方法
def getHitRatio(topk_acts, pos_act):
    if pos_act in topk_acts:
        return 1
    else:
        return 0

# 彩蝶程序里面的方法
def getNDCG(topk_acts, pos_act):
    for i in range(len(topk_acts)):
        act = topk_acts[i]
        if act == pos_act:
            return math.log(2) / math.log(i + 2)
    return 0

# cutoff = 1，需要搜索到第cutoff 跳邻居
# 参加活动越多的用户结点越大（size授予最大）
#参加一次活动，结点大小增加加size_rate
#要挑选的陪衬结点数据，即第9类结点
def getGraphDict(groupNum, maxsize, size_rate, anotherNodeNum, test_loader, model):
    #全局信息加载
    # categories = [{"name": "群体A"}, {"name": "群体B"}, {"name": "群体C"}, {"name": "群体D"}, {"name": "群体E"},
    #               {"name": "群体F"}, {"name": "群体G"}, {"name": "群体H"}, {"name": "邻居I"}]
    categories = []
    for i in range(0, groupNum):
        categories.append({'name': '群体' + str(i+1)})
    categories.append({'name': '群体邻居'})

    # 读入图结构
    graphPath = os.path.dirname(os.path.dirname(__file__)) + '/static/data/groupBehaviorPrediction/yelp_m10a3/graph.txt'
    pd_edges = pd.read_csv(graphPath, sep=' ', header=None, names=['src', 'tar'])
    pd_edges['src'] = pd_edges['src'].map(int)
    pd_edges['tar'] = pd_edges['tar'].map(int)
    graphEdges = pd_edges.values.tolist()
    GroupGraph = nx.Graph(graphEdges)

    # 读入用户-活动结构，单人最多参加1167次活动
    filename2 = os.path.dirname(
        os.path.dirname(__file__)) + '/static/data/groupBehaviorPrediction/yelp_m10a3/user_events.dat'
    user_acts = pd.read_csv(filename2, sep='\t', header=None, names=['user', 'acts'])
    userActList = user_acts.values.tolist()

    #读入用户-群体列表，单人参与群体的列表，组成的字典集合
    filename3 = os.path.dirname(os.path.dirname(__file__)) + '/static/data/groupBehaviorPrediction/yelp_m10a3/userid_groupid.json'
    with open(filename3, "r", encoding='utf-8') as jsonData:
        user_groups = json.load(jsonData)
    #返回数据样式构造
    GraphDict = {}  # 该群体封装的字典格式（网络信息）
    nodes = []  # 前端展示需要封装的列表格式（结点信息）
    links = []  # 前端展示需要封装的列表格式（边结构信息）
    # '''获取8个群体的成员，并封装出对应的结点集合'''
    #群体集合
    groupList = []
    #结点集合
    nodeList = []
    # 群体事件采样集合
    groupActSampleSet = []
    #群体事件预测结果
    groupActsPredict = []

    #获取x号群体的封装信息，添加进groupList内，x表示第几号群体（挑选第x次）
    def get_X_GroupInfo(groupList, x):
        # 随机取x号群体信息
        groupidX, membersX, pos_act, neg_acts, act_seq_index = getAGroup(groupList, test_loader, model.group_members)
        groupList.append(groupidX)
        #群体事件预测算法模块（评估）
        act_inputs = torch.LongTensor(neg_acts + [pos_act])
        group_inputs = torch.LongTensor([groupidX] * len(act_inputs))
        act_seq_ixs = torch.LongTensor([act_seq_index] * len(act_inputs))
        model.eval()
        pred_y = model(group_inputs, act_inputs, act_seq_ixs)
        #以下为求命中率，调用彩蝶的方法
        act_scores = {}
        for act, score in zip(act_inputs, pred_y.view(-1)):
            act_scores[act.item()] = score.item()
        # 计算每种K值下的HR和NDCG
        hr_k = []
        ndcg_k = []
        topk_acts = heapq.nlargest(15, act_scores, key=act_scores.get)
        hr = getHitRatio(topk_acts, pos_act)
        # ndcg = getNDCG(topk_acts, pos_act)
        hr_k.append(hr)
        # ndcg_k.append(ndcg)

        mainNodeSetX, nodeSetX = groupDisplayInfo(userActList, maxsize, size_rate, groupidX, membersX, x)
        #返回已经添加的groupList，已经添加了的总nodeList[去除重复]，x号群体含有的主体结点mainNodeSetX，前端格式封装好的nodeSetX
        return groupList, mainNodeSetX, nodeSetX, act_inputs, pred_y, hr_k

    # 随机取0~7号群体信息
    # groupList, mainNodeSet0, nodeSet0 = get_X_GroupInfo(groupList, 0)
    # groupList, mainNodeSet1, nodeSet1 = get_X_GroupInfo(groupList, 1)
    # groupList, mainNodeSet2, nodeSet2 = get_X_GroupInfo(groupList, 2)
    # groupList, mainNodeSet3, nodeSet3 = get_X_GroupInfo(groupList, 3)
    # groupList, mainNodeSet4, nodeSet4 = get_X_GroupInfo(groupList, 4)
    # groupList, mainNodeSet5, nodeSet5 = get_X_GroupInfo(groupList, 5)
    # groupList, mainNodeSet6, nodeSet6 = get_X_GroupInfo(groupList, 6)
    # groupList, mainNodeSet7, nodeSet7 = get_X_GroupInfo(groupList, 7)
    HRs = []
    # NDCGs = []
    groupNodeList = []
    # 随机取0~groupNum号群体信息
    for i in range(0, groupNum):
        groupList, mainNodeSetX, nodeSetX, act_inputs, temp_pred_y, hr_k = get_X_GroupInfo(groupList, i)
        groupNodeList.append(nodeSetX)
        act_inputsN = [int(tensor.numpy()) for tensor in act_inputs]
        temp_pred_yN = [round(float(tensor.detach().numpy()), 3) for tensor in temp_pred_y]
        groupActSampleSet.append(act_inputsN)
        groupActsPredict.append(temp_pred_yN)
        HRs.append(hr_k[:])
        # NDCGs.append(ndcg_k[:])

    HR_mean = torch.tensor(HRs, dtype=torch.float).mean(dim=0)
    # NDCG_mean = torch.tensor(NDCGs, dtype=torch.float).mean(dim=0)
    # print(float(HR_mean[0].detach().numpy()))
    # print(float(NDCG_mean[0].detach().numpy()))

    #将每个群体提取的封装信息融合，去除重叠的结点（即出现在多个群体内的结点），最终整理出前端展示需要的格式
    # nodes = nodeSet0
    nodes = groupNodeList[0]
    #将每个群体提取的封装信息融合，去除重叠的结点（即出现在多个群体内的结点），最终整理出前端展示需要的格式
    #nodes是挑选出来的结点信息（前端展示封装样式），nodeList是已经出现过的结点id集合，nodeSetX是当前群体内结点id的集合
    def groupNodesAddInNodesInfo(nodesCopy, nodeSetX):
        #获取当前已经加入的结点id集合
        nodeList = getNodeIdListFromNodesList(nodesCopy)
        for tempNode in nodeSetX:
            if int(tempNode['id']) not in nodeList:
                nodesCopy.append(tempNode)
        return nodesCopy

    # 去除0~n号群体重复的结点信息，前端展示结点不能有重复id，但是用户结点会参与多个群体
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet1)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet2)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet3)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet4)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet5)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet6)
    # nodes = groupNodesAddInNodesInfo(nodes, nodeSet7)
    # 去除0~groupNum号群体重复的结点信息，前端展示结点不能有重复id，但是用户结点会参与多个群体
    for i in range(0, groupNum):
        nodes = groupNodesAddInNodesInfo(nodes, groupNodeList[i])

    # 群体主体结点已经添加，剩下要进行递归搜寻邻居
    # 目前几个群体已经添加进去的结点id集合
    groupNodeIdList = getNodeIdListFromNodesList(nodes)
    all_neighbors = []
    for mainNode in groupNodeIdList:
        my_neighbors = nx.all_neighbors(GroupGraph, mainNode)
        all_neighbors = list(set(all_neighbors + list(my_neighbors)))
    #随机打乱群体邻居
    random.shuffle(all_neighbors)
    selectNum = 0
    for neighbourNodeId in all_neighbors:
        if selectNum >= anotherNodeNum:
            break
        if neighbourNodeId not in groupNodeIdList:
            x, y, categoryName = getLocationByCategory3(groupNum)
            tempNode = {}
            tempNode['id'] = str(neighbourNodeId)
            tempNode['name'] = str(neighbourNodeId)
            # 应该改成这个人参加活动的多少增大结点
            tempNode['symbolSize'] = (1 + getActNumByNodeId(userActList, int(neighbourNodeId)) * size_rate) % maxsize
            tempNode['x'] = x
            tempNode['y'] = y
            # 根据结点id找到[非现有群体内的，归属群体id]，负值表示每找到对应的群体
            tempNode['value'] = int(getGroupIdByNoedId2(groupList, str(neighbourNodeId), user_groups))
            tempNode['category'] = groupNum
            groupNodeIdList.append(neighbourNodeId)
            nodes.append(tempNode)
            selectNum = selectNum + 1

    # 利用networkx里面的算法自动生成位置，更新位置布局
    groupNodeIdList = getNodeIdListFromNodesList(nodes)
    subGraph = GroupGraph.subgraph(groupNodeIdList)
    #获取位置
    # 记录每个节点的位置信息
    pos = nx.drawing.spiral_layout(subGraph)
    pos = nx.drawing.spring_layout(subGraph, iterations=50, k=0.5)
    for node in nodes:
        id = node['id']
        node['x'] = pos[int(id)][0]
        node['y'] = pos[int(id)][1]
    # 挑选这8波群体的k阶邻居作为陪衬结点进行展示（避免结点都一个size过于单调）
    # k跳邻居字典
    # k_cutoff_nodes_dict = {}
    # for mainNode in groupNodeIdList:
    #     k_cutoff_nodes_dict[str(mainNode)] = nx.single_source_shortest_path_length(GroupGraph, int(mainNode), cutoff)
    # #邻居结点的，封装引入
    # if cutoff > 0:
    #     #方案一，所有非重叠结点全部读入，事实是结点太多，展示不下来
    #     # for key, value in k_cutoff_nodes_dict.items():
    #     #     for tempKey, tempValue in value.items():
    #     #         if tempKey not in groupNodeIdList:
    #     #             x, y, categoryName = getLocationByCategory(8)
    #     #             tempNode = {}
    #     #             tempNode['id'] = str(tempKey)
    #     #             tempNode['name'] = str(tempKey)
    #     #             # tempNode['symbolSize'] = (getActNumByNodeId(userActList, int(tempKey)) * size_rate) % maxsize#应该改成这个人参加活动的多少增大结点
    #     #             tempNode['symbolSize'] = str(random.randint(1, 25))  # 调整大小
    #     #             tempNode['x'] = x
    #     #             tempNode['y'] = y
    #     #             #根据结点id找到[非现有群体内的，归属群体id]，负值表示每找到对应的群体
    #     #             # tempNode['value'] = 'Group ID:' + str(getGroupIdByNoedId2(groupList, str(tempKey), user_groups))#效率太低，得改进查找效率
    #     #             tempNode['value'] = int(tempValue)#k跳邻居
    #     #             tempNode['category'] = 8
    #     #             groupNodeIdList.append(tempKey)
    #     #             nodes.append(tempNode)
    #     #方案二，随机筛选100个结点
    #     selectNum = 0
    #     len1 = len(k_cutoff_nodes_dict)
    #     neighbourDict = list(k_cutoff_nodes_dict.values())
    #     #随机采样，取邻居方法，有bug，可能样例少于要采取的邻居个数，永远采不出那么多结点
    #     while selectNum <= anotherNodeNum:
    #         index1 = random.randint(0, len1-1)
    #         cutoff_nodes = neighbourDict[index1]
    #         index2 = random.randint(1, len(cutoff_nodes)-1)
    #         neighbourList = list(cutoff_nodes.keys())
    #         # cutOffList = list(cutoff_nodes.values())
    #         neighbourNodeId = neighbourList[index2]
    #         # cut_off = cutOffList[index2]
    #         if neighbourNodeId not in groupNodeIdList:
    #             x, y, categoryName = getLocationByCategory3(groupNum)
    #             tempNode = {}
    #             tempNode['id'] = str(neighbourNodeId)
    #             tempNode['name'] = str(neighbourNodeId)
    #             tempNode['symbolSize'] = (getActNumByNodeId(userActList, int(neighbourNodeId)) * size_rate) % maxsize#应该改成这个人参加活动的多少增大结点
    #             tempNode['x'] = x
    #             tempNode['y'] = y
    #             #根据结点id找到[非现有群体内的，归属群体id]，负值表示每找到对应的群体
    #             tempNode['value'] = int(getGroupIdByNoedId2(groupList, str(neighbourNodeId), user_groups))#效率太低，得改进查找效率，暂时没改
    #             tempNode['category'] = groupNum
    #             groupNodeIdList.append(neighbourNodeId)
    #             nodes.append(tempNode)
    #             selectNum = selectNum + 1

    #根据构建的network，去封装并添加前端需要展示的边集合信息

    #封装边信息
    for e in subGraph.edges():
        link = {}
        link['source'] = str(e[0])
        link['target'] = str(e[1])
        links.append(link)

    # print(len(groupList))
    # print(len(groupActSampleSet))
    # print(len(groupActsPredict))
    # print(groupActsPredict[0])
    #开始按照样例格式封装数据的三个部分返回给调用者
    return nodes, links, categories, groupList, groupActSampleSet, groupActsPredict, float(HR_mean[0].detach().numpy())

# #挑选的群体个数
# groupNum = 5
# # 参加活动越多的用户结点越大（size授予最大）
# maxsize = 60
# #参加一次活动，结点大小增加加size_rate
# size_rate = 0.05
# #要挑选的陪衬结点数据，即第9类结点
# anotherNodeNum = 50
# starttime = datetime.datetime.now()
# nodes, links, categories = getGraphDict(groupNum, maxsize, size_rate, anotherNodeNum)
# endtime = datetime.datetime.now()
# print(endtime - starttime)

# #获取位置
# # 记录每个节点的位置信息
# pos = nx.drawing.spring_layout(GroupGraph, iterations=100, k=0.5)
# node_coordinate = []
# for i in range(GroupGraph.number_of_nodes()):
#     node_coordinate.append([])
# for i, j in pos.items():
#     # node_coordinate[node_dict[i]].append(float(j[0]))
#     # node_coordinate[node_dict[i]].append(float(j[1]))
#     node_coordinate[i - 1].append(float(j[0]))
#     node_coordinate[i - 1].append(float(j[1]))