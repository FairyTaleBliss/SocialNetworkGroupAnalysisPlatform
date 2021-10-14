from flask import Flask, render_template, request
import json
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
    app.run()
    # server = pywsgi.WSGIServer(('127.0.0.1', 5000), app)
    # server.serve_forever()
