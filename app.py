from flask import Flask, render_template, request
import json
import os
import sys
import datetime
import torch
import numpy as np
from algorithm import prediction
from algorithm.groupBehaviorPrediction.DataLoader import Dataloader
from algorithm.groupBehaviorPrediction.LSTMGCNPMAgbp import LSTMGCNPMAGbp
from gevent import pywsgi

app = Flask(__name__)

#主界面跳转服务
@app.route('/')
def index():
    return render_template('index.html')

#登录后台服务
@app.route('/login')
def login():
    return render_template('login.html')

# 获得已登录用户信息服务（暂时写死，后期改成读取数据库用户信息表）
@app.route('/getUserInfo', methods=["POST"])
def getUserInfo():
    userInfo = {}
    userInfo['user'] = 'test user'
    userInfo['phoneNumber'] = '12345678910'
    userInfo['mail'] = 'example@shu.com'
    userInfo['unit'] = '上海大学'
    userInfo['isSuccess'] = 1
    newData = json.dumps(userInfo)  # json.dumps封装
    return newData

# 使用说明，内嵌子页面跳转服务
@app.route('/introduce')
def introduce():
    '''
    TODO
    :return:
    '''
    test = "使用说明界面"
    # 把需要的数据给对应的页面
    return render_template('introduce.html', test=test)

# 关于我们，内嵌子页面跳转服务
@app.route('/aboutUs')
def aboutUs():
    '''
    TODO
    :return:
    '''
    return render_template('aboutUs.html')

# 微博转发结构分析，跳转服务
@app.route('/AnalysisOfWeiboForwardingStructure')
def AnalysisOfWeiboForwardingStructure():
    '''
    TODO
    :return:
    '''
    return render_template('AnalysisOfWeiboForwardingStructure.html')

# 单条微博情感分析，后端服务
@app.route('/EmotionalAnalysisOfSingleWeibo')
def EmotionalAnalysisOfSingleWeibo():
    '''
    TODO
    :return:
    '''
    return render_template('EmotionalAnalysisOfSingleWeibo.html')

# 微博情感整体分析，后端服务
@app.route('/OverallAnalysisOfWeiboSentiment')
def OverallAnalysisOfWeiboSentiment():
    '''
    TODO
    :return:
    '''
    return render_template('OverallAnalysisOfWeiboSentiment.html')

# 微博情感详细分析，后端服务
@app.route('/DetailedAnalysisOfWeiboSentiment')
def DetailedAnalysisOfWeiboSentiment():
    '''
    TODO
    :return:
    '''
    return render_template('DetailedAnalysisOfWeiboSentiment.html')

# 介绍SI模型，后端服务
@app.route('/introduceSIModel')
def introduceSIModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceSIModel.html')

# SI模型展示，后端服务
@app.route('/DemonstrationOfSIModel')
def DemonstrationOfSIModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfSIModel.html')

# 介绍SIR模型，后端服务
@app.route('/introduceSIRModel')
def introduceSIRModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceSIRModel.html')

# SIR模型展示，后端服务
@app.route('/DemonstrationOfSIRModel')
def DemonstrationOfSIRModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfSIRModel.html')

# 介绍谣言溯源模型，后端服务
@app.route('/introduceRumorTraceabilityModel')
def introduceRumorTraceabilityModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceRumorTraceabilityModel.html')

# 谣言溯源模型展示，后端服务
@app.route('/DemonstrationOfRumorTraceabilityModel')
def DemonstrationOfRumorTraceabilityModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfRumorTraceabilityModel.html')

# 谣言溯源模型比对，后端服务
@app.route('/ComparisonOfRumorTraceabilityModel')
def ComparisonOfRumorTraceabilityModel():
    '''
    TODO
    :return:
    '''
    return render_template('ComparisonOfRumorTraceabilityModel.html')

# 介绍群体行为预测模型，跳转服务
@app.route('/introduceGroupBehaviorPredictionModel')
def introduceGroupBehaviorPredictionModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceGroupBehaviorPredictionModel.html')

# 群体行为预测模型展示，后端服务
@app.route('/DemonstrationOfGroupBehaviorPredictionModel')
def DemonstrationOfGroupBehaviorPredictionModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfGroupBehaviorPredictionModel.html')

# 群体行为预测模型展示页面，请求数据
@app.route('/PredictionPageGetData', methods=["POST"])
def PredictionPageGetData():
    # 获取前端请求的数据
    step = request.form.get('step')
    stepNum = int(step)
    #需要封装传给前端的数据
    formData = {}
    #分阶段展示，第一阶段请求响应
    if stepNum == 0:
        #直接用样例数据
        # filename = os.path.dirname(__file__) + '/static/data/example.json'
        # with open(filename, "r", encoding='utf-8') as jsonData:
        #     jsonDataLoad = json.load(jsonData)
        #     formData['nodes'] = jsonDataLoad['nodes']
        #     formData['links'] = jsonDataLoad['links']
        #     formData['categories'] = jsonDataLoad['categories']
        # 挑选的群体数
        groupNum = 4
        # 需要搜索到几跳邻居
        # cutoff = 1
        # 参加活动越多的用户结点越大（size授予最大）
        maxsize = 60
        # 参加一次活动，结点大小增加加size_rate
        size_rate = 0.1
        # 要挑选的陪衬结点数据，即第9类结点
        anotherNodeNum = 50
        nodes, links, categories, groupList, groupActSampleSet, groupActsPredict, hitRate = prediction.getGraphDict(groupNum, maxsize, size_rate, anotherNodeNum, test_loader, model)
        formData['nodes'] = nodes
        formData['links'] = links
        formData['categories'] = categories
        formData['groupList'] = groupList
        formData['groupActSampleSet'] = groupActSampleSet
        formData['groupActsPredict'] = groupActsPredict
        formData['hitRate'] = hitRate
        formData['execute'] = 'success'
        return json.dumps(formData)
    elif stepNum == 1:
        #触发前端结构刷新按钮
        # 获取前端请求的数据
        groupNum = int(request.form.get('groupNum'))
        # cutoff = int(request.form.get('cutoff'))
        maxsize = int(request.form.get('maxSize'))
        size_rate = float(request.form.get('sizeRate'))
        anotherNodeNum = int(request.form.get('neighbourNumber'))
        nodes, links, categories, groupList, groupActSampleSet, groupActsPredict, hitRate = prediction.getGraphDict(groupNum, maxsize, size_rate, anotherNodeNum, test_loader, model)
        formData['nodes'] = nodes
        formData['links'] = links
        formData['categories'] = categories
        formData['groupList'] = groupList
        formData['groupActSampleSet'] = groupActSampleSet
        formData['groupActsPredict'] = groupActsPredict
        formData['hitRate'] = hitRate
        formData['execute'] = 'success'
        return json.dumps(formData)
        #处理成功，成功响应
        formData['execute'] = 'success'
        return json.dumps(formData)

    #不在展示阶段内的步骤，代表处理异常
    formData['execute'] = 'fail'
    return json.dumps(formData)  # json.dumps封装

# 介绍群体行为传播模型，跳转服务
@app.route('/introduceGroupBehaviorCommunicationModel')
def introduceGroupBehaviorCommunicationModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceGroupBehaviorCommunicationModel.html')

# 群体行为传播模型展示，后端服务
@app.route('/DemonstrationOfGroupBehaviorCommunicationModel')
def DemonstrationOfGroupBehaviorCommunicationModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfGroupBehaviorCommunicationModel.html')

# 介绍群体情感分析模型，跳转服务
@app.route('/introduceGroupSentimentAnalysisModel')
def introduceGroupSentimentAnalysisModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceGroupSentimentAnalysisModel.html')

# 群体情感分析模型展示，后端服务
@app.route('/DemonstrationOfGroupSentimentAnalysisModel')
def DemonstrationOfGroupSentimentAnalysisModel():
    '''
    TODO
    :return:
    '''
    return render_template('DemonstrationOfGroupSentimentAnalysisModel.html')

if __name__ == '__main__':
    # 群体预测模块，模型加载
    print('群体预测模块，模型加载开始。。。')
    starttime = datetime.datetime.now()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device('cpu')
    embed_size = 32
    batch_size = 256
    lr = 0.01
    lr_user = 0.05
    num_epoch = 10
    train_rate = 0.8  # 训练集比例
    num_negatives = 3
    seq_length = 20
    dropout = 0.2
    weight_decay = 1e-6
    K = [1, 3, 5, 10, 15]
    seed = 42
    pretrain_user = False

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    setup_seed(seed)  # 设置随机数种子

    print(
        'embed_size = {},batch_size = {},lr = {}, num_epoch = {}, seq_length = {}, num_negatives = {}, seed = {}, dropout = {}, weight_decay = {}' \
            .format(embed_size, batch_size, lr, num_epoch, seq_length, num_negatives, seed, dropout, weight_decay))
    filename = os.path.dirname(__file__) + '/static/data/groupBehaviorPrediction/yelp_m10a3'
    dataloader = Dataloader(filename)  # douban_g15000
    print('用户个数：', len(dataloader.users))
    print('群体个数：', len(dataloader.groups))
    print('act个数：', dataloader.act_num)
    print('节点个数：', dataloader.graph.number_of_nodes())
    train_loader, test_loader, all_act_seqs = dataloader.get_dataloader(batch_size, train_rate, num_negatives)
    print('\n群体行为数据加载完毕！')

    (train_act_seqs, test_act_seqs) = all_act_seqs
    for i in range(len(train_act_seqs)):
        g_act_seqs_tr = train_act_seqs[i]
        train_act_seqs[i] = [seq.to(device) for seq in g_act_seqs_tr]
    for i in range(len(test_act_seqs)):
        g_act_seqs_te = test_act_seqs[i]
        test_act_seqs[i] = [seq.to(device) for seq in g_act_seqs_te]

    for g in dataloader.group_members:
        dataloader.group_members[g] = dataloader.group_members[g].to(device)

    aggregators = ["mean", "max", "min", "std"]
    scalers = ["identity", "amplification", "attenuation"]

    # aggregators = ["mean"]
    print(aggregators, scalers)
    print('LSTMPMAGbp模型')

    model = LSTMGCNPMAGbp(num_act=dataloader.act_num,
                          num_user=len(dataloader.users),
                          embed_size=embed_size,
                          group_members=dataloader.group_members,
                          all_act_seqs=all_act_seqs,
                          graph=dataloader.graph,
                          aggregators=aggregators,
                          scalers=scalers,
                          dropout=dropout).to(device)

    model.load_state_dict(torch.load(os.path.dirname(__file__) + '/static/data/groupBehaviorPrediction/pma_yp.pt', map_location='cpu'))
    endtime = datetime.datetime.now()
    print('群体预测模块，模型加载结束！！！')
    print('加载用时：', endtime - starttime)
    # print(test_loader[0])
    # print(test_loader[1])
    # print(test_loader[2])

    #总运行
    app.run()
    # server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    # server.serve_forever()
