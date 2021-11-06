import pandas as pd
import random
import networkx as nx
import os
#所有用户初始情感值为0
workpath=os.path.dirname(os.path.dirname(__file__))


txtpath_init = workpath+'/static/data/init_dblp.csv'
def user_senti_init():
    init_value=0
    return init_value

def group_emotion_init():
    group_value=0
    return group_value

#计算用户的情感向量表示
def calsenti_vector(G):
    import os
    def check_if_dir(file_path, path_read):
        temp_list = os.listdir(file_path)  # put file name from file_path in temp_list
        for temp_list_each in temp_list:
            if os.path.isfile(file_path + '/' + temp_list_each):
                temp_path = file_path + '/' + temp_list_each
                if os.path.splitext(temp_path)[-1] == '.csv':  # csv文件加一个判断
                    path_read.append(temp_path)
                else:
                    continue
            else:
                check_if_dir(file_path + '/' + temp_list_each, path_read)  # loop traversal
        return path_read
    pathread=[]
    pathread = check_if_dir(workpath+'/static/data/dblp_weizao', pathread)
    # print(pathread)
    # alltxt='all.csv'
    alltxt=pd.read_csv(pathread[0])
    # alltxt
    for i in range(1,len(pathread)):
        content=pd.read_csv(pathread[i])
        # print(content)
        alltxt=pd.concat([alltxt,content],axis=0)
    alltxt=alltxt.drop_duplicates()
    # print(G.nodes())
    alltxt=alltxt.sort_values(by='user_id')
    alltxt.to_csv(workpath+'/static/data/dblp_all.csv',index=False)
    alltxt=pd.read_csv(workpath+'/static/data/dblp_all.csv')
    # print(alltxt)

    user_senti_vector={}
    for i in G.nodes:
        user_senti_vector[i]=[0,0,0]
    for i in range(len(alltxt)):
        if alltxt['user_id'][i] in G.nodes:
            if alltxt['senti_value'][i]==0:
                user_senti_vector[alltxt['user_id'][i]][1]+=1
            elif alltxt['senti_value'][i]>0:
                user_senti_vector[alltxt['user_id'][i]][0] += 1
            else:
                user_senti_vector[alltxt['user_id'][i]][2] += 1
    for key in user_senti_vector.keys():
        sum=user_senti_vector[key][0]+user_senti_vector[key][1]+user_senti_vector[key][2]
        user_senti_vector[key][0]=user_senti_vector[key][0]/sum
        user_senti_vector[key][1] = user_senti_vector[key][1] / sum
        user_senti_vector[key][2] = user_senti_vector[key][2] / sum
    return user_senti_vector

import math
import numpy as np
# 计算两点之间的欧式距离
def eucliDist(A,B):
    return math.sqrt(sum([(a - b)**2 for (a,b) in zip(A,B)]))

def generate_graph2():
    datasets = []  # 图
    data_file = open(os.path.dirname(os.path.dirname(__file__)) + '/static/data/dblp.txt', "r") # 读取文本
    data = data_file.read()
    rows = data.split('\n')
    for row in rows:
        # split_row = row.split('\t') #graph.txt数据
        split_row = row.split(' ')  # dblp.txt
        name = (int(split_row[0]), int(split_row[1]))  # random.random(0,1)作为不同边的影响传播概率，后期可以以用户文本的相似性进行替换，可以考虑主题分布
        datasets.append(name)
        #    print(datasets)
    G = nx.DiGraph()  # 将图设置为有向图格式
    G.add_edges_from(datasets)  # 对datasets元组的每个增加边
    for node in G:
        G.add_node(node, state=0)  # 用state标识状态 state=0 未激活，state=1 激活
    user_senti_vector = calsenti_vector(G)
    from scipy.stats import pearsonr
    outdegree = dict(G.out_degree)
    maxdegree = sorted(outdegree.items(), key=lambda item: item[1], reverse=True)
    mindegree=maxdegree[-1][1]
    maxdegree=maxdegree[0][1]
    for edge in G.edges:
        # 定义情感相似性作为阈值,余弦相似度的计算方式
        # wei=random.uniform(0,0.1)
        # wei=0.001
        # wei=1/G.in_degree(edge[1])
        #基于情感影响的传播模型，边阈值
        # a=0.5
        # a=0.1
        a=0.5
        b=0.1
        p1=(1+a)/(G.in_degree(edge[1])+a) #用户u和v之间收到消息的概率
        sim = np.dot(user_senti_vector[edge[0]], user_senti_vector[edge[1]]) / (
                    np.linalg.norm(user_senti_vector[edge[0]]) * np.linalg.norm(user_senti_vector[edge[1]])) #用户v接受用户u消息的概率
        inf_v=(G.out_degree(edge[1])-mindegree)/(maxdegree-mindegree)
        if sim >= 1/(1+b):
            b=(1-sim)/sim
        p2=(1+b*inf_v)*sim
        # print(p2)
        wei=p1*p2


        # wei = np.dot(user_senti_vector[edge[0]], user_senti_vector[edge[1]]) / (
        #             np.linalg.norm(user_senti_vector[edge[0]]) * np.linalg.norm(user_senti_vector[edge[1]]))
        # wei=eucliDist(user_senti_vector[edge[0]],user_senti_vector[edge[1]])
        # wei=pearsonr(user_senti_vector[edge[0]],user_senti_vector[edge[1]]) #采用皮尔逊相关系数
        G.add_edge(edge[0], edge[1], weight=wei)  # 增加权值 采用皮尔逊相关系数时需要求绝对值，即abs(wei[0])，其他概率下只需要wei即可
    # 求解所有用户的入度节点，并保存
    user_in={}#保存用户的入度节点
    for edge in G.edges:
        if edge[1] not in user_in.keys():
            user_in[edge[1]]=[]
            user_in[edge[1]].append(edge[0])
        else:
            user_in[edge[1]].append(edge[0])
    return G,user_in

def group_emotion(txtpath,txtpath_last,group,period_,G,user_in):
    #txtpath_last 上一时刻，txtpath 当前时刻
    txt = pd.read_csv(txtpath)
    userid = set(txt['user_id'])
    group_value = 0
    period= {}
    for i in G.nodes():
        period[i] = 0
        if i in userid:
            usersenti=user_senti_period(i, txtpath_last, txtpath,group,period_,G,user_in)
            period[i]=usersenti #保存每个阶段的用户情感值排序（结合了群体情感与结构属性）
        group_value += period[i]
        # print(i, group_value)
    # period.sort(key=lambda x: x[1], reverse=True)
    group_value =1/(1+math.exp(-group_value/len(userid)))
    return period,group_value

#邻居上一时刻的情感值对用户现在的影响
def inneigborvalue(user_id,txtpath_last,period,G,user_in):
    if txtpath_last==txtpath_init:
        neigbborvalue=0
    if txtpath_last!=txtpath_init:
        neigbborvalue = 0
        if user_id in user_in.keys():
            inneigh=user_in[user_id]
            # print(inneigh)
            if len(inneigh)!=0:
                for i in inneigh:
                    if G.out_degree(i)!=0:
                    # print(period[txtpath_last])#[1, 23, 4, 6, 17, 21, 2, 8, 3, 5, 7, 9, 13, 10, 11, 12, 14, 15, 16, 18, 19, 20, 22]
                        neigbborvalue+=period[txtpath_last][i]/G.out_degree(i)
                    else:
                        neigbborvalue += period[txtpath_last][i]
    return neigbborvalue

#用户的结构属性，采用Jaccard相似度
def structure_feature(G):
    # G,user_in=generate_graph2()
    for edge in G.edges:
        ngb1=list(G[edge[0]])
        ngb2=list([edge[1]])
        # print(ngb1)
        jiaoji=set(ngb1).intersection(set(ngb2))
        bingji=set(ngb1).union(set(ngb2))
        stru_fea=len(jiaoji)/len(bingji)
        # print(stru_fea)
        # if G[edge[0]][edge[1]]['weight']*stru_fea>1:
        # print(G[edge[0]][edge[1]]['weight'],stru_fea,G[edge[0]][edge[1]]['weight']*stru_fea)
        G[edge[0]][edge[1]]['weight']=G[edge[0]][edge[1]]['weight']*stru_fea
        # print(G[edge[0]][edge[1]]['weight'])
    return G



def user_senti_period(user_id,txtpath_last,txtpath,group,period,G,user_in):
    # print(group)
    group_value_last = group[txtpath_last]
    txt=pd.read_csv(txtpath)
    value_own=0 #个人情感值
    value_pos = 0  # 个人情感值
    value_neg = 0  # 个人情感值
    value_neu = 0  # 个人情感值

    # usernum = 955
    # sumago=sum(txt['dianzan']) + sum(txt['zhuanfa']) #当前阶段所有用户转发和点赞的数量和
    # sum1 = sumago*usernum/70000 #按照比例缩减的点赞和转发数目,现在，usernum是当前所有人数，70000是爬取的数目用户之和
    # print(sum1)
    min_value=min(txt['senti_value'])
    max_value=max(txt['senti_value'])
    count=0
    for i in range(len(txt)):#用户在该时间段下所有文本的情感状态和，考虑结构属性
        if user_id == txt['user_id'][i]:
            # print(txt['senti_value'][i])
            count+=1
            # value=(txt['senti_value'][i] - min_value) / (max_value - min_value) #采用离差标准化对数据进行归一化，防止情感值一直增加,对原始数据的线性变换，使结果值映射到[0 - 1]之间
            # print(value)
            if txt['senti_value'][i]>0:
                # value_own += 1*1
                value_pos += 1
                # value_own+=value*(1+(txt['dianzan'][i]+txt['zhuanfa'][i])*sum1/sumago) #用户本身情感值+对他人的影响情感值
            elif txt['senti_value'][i]<0:
                # value_own += (-1)*value * (1 + (txt['dianzan'][i] + txt['zhuanfa'][i])*sum1/sumago)  # 用户本身情感值+对他人的影响情感值
                # value_own += (-1) * 1
                value_neg += 1
            else:
                # value_own +=0*1
                value_neu += 1
    # print( count,value_pos,value_neu,value_neg)
    if value_neg==0:
        if value_neu == 0:
            value_own=1
        elif value_pos == 0:
            value_own = 1
        else:
            value_own=1*value_pos/count*math.log(count/value_pos)+0*value_neu/count*math.log(count/value_neu)
    elif value_neu==0:
        if value_pos == 0:
            value_own=1
        elif value_neg==0:
            value_own = 1
        else:
            value_own=1*value_pos/count*math.log(count/value_pos)+(-1)*value_neg/count*math.log(count/value_neg)
    elif value_pos==0:
        if value_neg == 0:
            value_own=1
        elif value_neu == 0:
            value_own = 1
        else:
            value_own=(-1)*value_neg/count*math.log(count/value_neg)+0*value_neu/count*math.log(count/value_neu)
    else:
        value_own = (-1) * value_neg / count * math.log(count / value_neg) + 0 * value_neu / count * math.log(
            count / value_neu)+1*value_pos/count*math.log(count/value_pos)
    a1=0.5
    a2=1-a1
    # print(value_own)
    # b1=random.uniform(0.001,0.01)#用户接受群体影响的概率
    # if G.in_degree(user_id) !=0:
    #     b1=1/G.in_degree(user_id)#用户接受群体影响的概率
    # else:
    #     b1=1
    # if G.in_degree(user_id)!=0:
    #     b1=1/G.in_degree(user_id)  #用户接受邻居影响的概率
    # else:
    #     b1=random.random()
    b1 = random.uniform(0.01,0.1)#用户对其粉丝的影响概率
    b3=random.uniform(0.001,0.01) #用户接受上一时刻邻居对其的影响概率
    b2=random.uniform(0.01,0.1) #粉丝接受用户影响的概率
    # user_value=a1*value_own/count+a2*b1*b2*inneigborvalue(user_id,txtpath_last,period,G,user_in)
        #该用户在某一时间段下的平均情感状态，考虑邻居上一时刻对其的影响
    user_value=value_own*(1+b3*inneigborvalue(user_id,txtpath_last,period,G,user_in))*(a1/count+a2*b1*b2*G.out_degree(user_id))#用户对这个群体的情感贡献值
    return user_value

def icmodel(G, S, itnum):  # 输入网络图G，种子节点集，激活概率阈值，蒙托卡罗模拟次数，输出种子集合的影响力
    import numpy as np
    # print(G.nodes)
    spread = []
    # print(G.out_degree[1])
    for i in range(0, itnum):
        newactive, A = S[:], S[:]  # A记录一次迭代的所有激活节点集合
        # print(newactive)
        while newactive:
            newones = []
            for node in newactive:
                # print(list(G[node]))  # 表示当前节点的所有出度节点
                # print(np.random.uniform(0,1,G.out_degree[node]))
                # print(G[node])#表示当前节点的所有出度节点，
                for ngb in list(G[node]):
                    # print(G[node][ngb]['weight'])
                    # print(G[node][ngb]['weight'] )
                    if random.uniform(0,1) < G[node][ngb]['weight'] :
                        newones.append(ngb)
                #         success = np.random.uniform(0, 1, G.out_degree[node]) < p  # 按照当前节点的出度邻居数目生成相应的节点概率
                # # print(np.extract(success, G[node]))
                # newones +=list(np.extract(success, G[node]))
            newactive = list(set(newones)-set(A))
            A += newactive
        spread.append(len(A))
        # print('迭代',i,'次影响结点',A)
    return np.mean(spread)




