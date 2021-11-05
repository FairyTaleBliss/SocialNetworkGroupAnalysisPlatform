from flask import Flask, render_template, request
import json
from gevent import pywsgi
import os

app = Flask(__name__)

# 主界面跳转服务
@app.route('/')
def index():
    return render_template('index.html')

# 登录后台服务
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
@app.route('/DemonstrationOfGroupBehaviorCommunicationModel', methods=["GET", "POST"])
def DemonstrationOfGroupBehaviorCommunicationModel():
    '''
    TODO
    :return:
    '''
    import numpy as np
    import json
    # 读取数据
    networkTemp = []  # txt文件内前两节列
    nodes_20 = []  # txt文件内所有数据
    networkFile = open(os.path.dirname(__file__) + '/static/data/nodes_20.txt', 'r')

    for line in networkFile.readlines():
        linePiece = line.split(',')
        networkTemp.append([int(linePiece[0]), int(linePiece[1])])
        nodes_20.append([int(linePiece[0]), int(linePiece[1]), int(linePiece[2])])

    # 对networkTemp中的节点进行计数、显示
    # for i in range(344):
    #     for j in range(2):
    #         print(networkTemp[i][j])

    # 设置传给前端的节点数据边数据的json串
    graph_data_json = {}
    nodes_data_json = []

    # 获取所有应该显示的节点
    display = []
    source_nodes = []  # 存储所有的源节点
    # for displaynodes in range(number_of_nodes):
    #     for number in range(344):
    #         if displaynodes in networkTemp[number]:
    #             display.append(displaynodes)
    # display = list(set(display))
    # print(display)
    for i in range(len(networkTemp)):
        for j in range(len(networkTemp[1])):
            if networkTemp[i][j] in display:
                continue
            else:
                display.append(networkTemp[i][j])
                # source_nodes.append(networkTemp[i][0])

    for i in range(len(networkTemp)):
        # for j in range(len(networkTemp[1])):
        if networkTemp[i][0] in source_nodes:
            continue
        else:
            # display.append(networkTemp[i][j])
            source_nodes.append(networkTemp[i][0])

    for i in range(len(display)):
        node = display[i]
        nodes_data_json.append({})
        nodes_data_json[i]['attributes'] = {}
        nodes_data_json[i]['attributes']['modularity_class'] = 0
        nodes_data_json[i]['id'] = str(node)
        nodes_data_json[i]['category'] = 0
        nodes_data_json[i]['itemStyle'] = ''
        nodes_data_json[i]['label'] = {}
        nodes_data_json[i]['label']['normal'] = {}
        nodes_data_json[i]['label']['normal']['show'] = 'false'
        nodes_data_json[i]['name'] = str(node)
        nodes_data_json[i]['symbolSize'] = 35
        nodes_data_json[i]['value'] = 15
        nodes_data_json[i]['x'] = 0
        nodes_data_json[i]['y'] = 0

    links_data_json = []
    for link in networkTemp:
        links_data_json.append({})
        links_data_json[len(links_data_json) - 1]['id'] = str(len(links_data_json) - 1)
        links_data_json[len(links_data_json) - 1]['lineStyle'] = {}
        links_data_json[len(links_data_json) - 1]['lineStyle']['normal'] = {}
        links_data_json[len(links_data_json) - 1]['name'] = 'null'
        links_data_json[len(links_data_json) - 1]['source'] = str(link[0])
        links_data_json[len(links_data_json) - 1]['target'] = str(link[1])

    graph_data_json['nodes'] = nodes_data_json
    graph_data_json['links'] = links_data_json
    graph_data = json.dumps(graph_data_json)

    isOrigin = False
    # 存放所有节点id的列表
    nodes_show = []
    # 不显示所有群体
    show_group = False
    # 要显示的群体类别。为1时，表示前端展示群体1
    show_group_id = 0
    # 源节点的id
    source_node_id = 0
    # print(nodes_20)

    # source_nodes = list(set(source_nodes))  # 去重复
    # source_nodes.sort()
    post_node_id = request.form.get("source_id")
    if request.method == 'POST':
        # 获得选择框的内容
        select_node = request.form.get("select_node")
        if select_node == "全部":
            # 渲染全部的！
            show_group = True
        else:
            # 渲染指定的！
            show_group = False
            show_group_id = source_nodes.index(int(select_node))

        # 先判断是不是源节点或者空值，是空就返回0列表；不是源节点就返回值为-1的列表；不是就返回要显示的节点列表

        # def not_exist(node_id):  # 判断是不是源节点，不是就返回ture
        #     for i in range(len(source_nodes)):
        #         if source_nodes[i] == int(node_id):
        #             return False
        #     return True

        def nodes_20_index(node_index):  # 求源节点在nodes_20文件中第一次出现的索引值
            for i in range(len(nodes_20)):
                if nodes_20[i][0] == source_nodes[node_index]:  # 判断源节点列表中第node_index个源节点在nodes_20的列表中的索引
                    return i

        def nodes_20_end_index(node_index):  # 求第node_index个源节点在nodes_20文件中最后出现的索引值
            end_index = 0
            for i in range(len(nodes_20)):
                if nodes_20[i][0] == source_nodes[node_index]:
                    end_index = i
            return end_index

        # print(source_node_id)
        # print(nodes_20)

        # print(not_exist())

        def time_max(node_index):  # 求第node_index个源节点的群体里，最后一个扩散节点显示的时间
            t_max = nodes_20[nodes_20_end_index(node_index)][2]
            return t_max

        # if not post_node_id:  # 如果前端传来的id值是空的
        #     show_group = True  # 按顺序展示所有群体的节点

            # print(display)
        # 求所有群体的节点显示列表，形如[ [[964],[1034,341,1130,386,...],[],[],[320],[40,42,1014],[62]], [[1085],[1127,133,...],...], [...], ..... ]
        for i in range(len(source_nodes)):  # 遍历所有的源节点
            nodes_show_tmp = []
            list_source = [source_nodes[i]]  # 先将i源节点放在一个列表里
            nodes_show_tmp.append(list_source)  # 将i源节点列表添加到nodes_show列表的第i个子列表
            for j in range(1, time_max(i) + 1):  # 遍历i源节点代表的群体的 所有时间点
                list_spread = []
                for k in range(nodes_20_index(i), nodes_20_end_index(i) + 1):  # 遍历i源节点在nodes_20文件中所有的行
                    if nodes_20[k][2] == j:  # 判断i源节点的扩散节点的时间点是不是j
                        list_spread.append(nodes_20[k][1])  # 将i源节点在j时刻的扩散节点加入list_spread列表
                nodes_show_tmp.append(list_spread)  # 将list_spread添加到nodes_show的第i个子列表
            nodes_show.append(nodes_show_tmp)

        # else:
            # if not_exist(post_node_id):  # 如果前端传来的id值不是源节点
                # nodes_show.append(-1)
                # isOrigin = True  # 弹窗
            # else:  # 前端传来的id值是源节点，那么就展示show_group_id这个群体，
                # show_group_id=0表示：这个群体的源节点在source_nodes列表中的索引为0，这个群体所有要展示的节点在nodes_show[0]，这个群体在前端指的是群体1
                # show_group_id = source_nodes.index(int(post_node_id))
            #     # source_node_id = int(source_node_id)
            #     list_source = [int(source_node_id)]
            #     nodes_show.append(list_source)  # 将源节点加入列表
            #     for j in range(1, time_max() + 1):
            #         list_spread = []
            #         for k in range(nodes_20_index(), nodes_20_end_index() + 1):
            #             if nodes_20[k][2] == j:
            #                 list_spread.append(nodes_20[k][1])  # 将i时刻的扩散节点加入list
            #         nodes_show.append(list_spread)
        # print(nodes_20_index())
        # print(nodes_20_end_index())
        # print(time_max())
        # print(source_node_id)
    nodes_20 = json.dumps(nodes_20)
    display = json.dumps(display)
    # source_nodes = json.dumps(source_nodes)
    # print(nodes_show)
    # print(source_nodes)
    # print(show_group_id)

    return render_template('DemonstrationOfGroupBehaviorCommunicationModel.html',
                           graph_data=graph_data,
                           nodes_20=nodes_20,
                           display=display,
                           source_nodes=source_nodes,
                           alert=isOrigin,
                           show_group=show_group,
                           show_group_id=show_group_id,
                           nodes_show=nodes_show)


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
