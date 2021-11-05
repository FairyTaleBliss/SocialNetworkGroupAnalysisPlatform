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
import sqlite3
from gevent import pywsgi
import os
import os

# 微博可视化部分添加的库
import pandas as pd
import snownlp as sn
from snownlp import sentiment
import jieba
import wordcloud
import re
import matplotlib.pyplot as plt
import glob
import imageio
from wordcloud import WordCloud, ImageColorGenerator
from snownlp import sentiment
from PIL import Image, ImageDraw, ImageFont
from os import path
import math as m
import networkx as nx

# 谣言可视化部分添加的库
from flask import jsonify
from algorithm import gameTheory
from algorithm import sourceDetection as sd
from algorithm import SIModel as si
from algorithm import SIRModel as sir
from algorithm import opinionEvolution as oe
from algorithm import hawkesProcess
from algorithm import GE_sourceDetection as GE
import random
import copy
from flask_socketio import SocketIO
from flask_mail import Mail, Message


# 微博可视化部分支撑
data_path = 'static/data/weibo'
data_path_cache = 'static/data/weibo/analysis_cache'

def load_wb_data(path):
    weibo = pd.read_csv(path)
    #weibo['user'] = weibo['user'].map(eval)  # 将读取到的str数据转为dict类型
    num_wb = len(weibo)
    return weibo, num_wb

def load_ur_data(path):
    user = pd.read_csv(path)
    num_ur = len(user)
    return user, num_ur

class Repost():
    def __init__(self, path):
        self.reposts = self.load_repost_data(path)
        self.num_reposts = len(self.reposts)  # 转发微博总数量
        self.src_wb = self.reposts.iloc[0,]  # 源微博
        self.network = self.get_network()  # 网络结构
        self.post_indexs = {str(self.src_wb['id']): 0}  # 每条微博的编号
        self.coordinate = {str(self.src_wb['id']): {'x': 0, 'y': 0}}  # 节点坐标
        self.node_size = {str(self.src_wb['id']): 35}  # 节点大小
        self.category = {str(self.src_wb['id']): 0}  # 节点类别
        self.st_category = {}  # 节点情感类别
        self.num_category = 1  # 类别个数
        self.calc_all_node_cor()  # 计算所有节点的坐标
        self.graph_data = self.get_graph_data()  # 传给前端的图数据

    # 读取转发数据
    def load_repost_data(self, path):
        reposts = pd.read_csv(path)
        reposts['user'] = reposts['user'].map(eval)
        return reposts

    # 获取网络结构数据
    def get_network(self):
        network = []
        for i in range(1, self.num_reposts):
            post = self.reposts.iloc[i,]
            link = [str(post['pidstr']), str(post['id'])]
            if link not in network:
                network.append(link)
        return nx.DiGraph(network)

    # 获取传给前端的图数据
    def get_graph_data(self):
        nodes = [{  # 源微博节点
            # 'attributes': {'modularity_class': 0},
            'id': str(self.src_wb['id']),
            'category': self.category[str(self.src_wb['id'])],
            'itemStyle': '',
            # 'label': {'normal': {'show': 'false'}},
            'label': {'show': 'false'},
            'name': str(self.src_wb['user']['screen_name']),
            'symbolSize': self.node_size[str(self.src_wb['id'])],
            'value': self.src_wb['text'],
            'x': self.coordinate[str(self.src_wb['id'])]['x'],
            'y': self.coordinate[str(self.src_wb['id'])]['y']
        }]
        div_txt = clearTxt(str(self.src_wb['text']))
        st_cat = 2
        if div_txt:
            senti_value = sen_value(div_txt)
            if senti_value > 0.57:
                st_cat = 1
            elif senti_value < 0.5:
                st_cat = 0
        self.st_category.update({str(self.src_wb['id']):st_cat})
        links = []
        cur_nodes = []
        cur_links = []
        cur_index = 1
        for i in range(1, self.num_reposts):
            post = self.reposts.iloc[i,]  # 第i条转发微博
            node = str(post['id'])
            if node not in cur_nodes:
                self.post_indexs.update({node: cur_index})
                cur_index += 1
                # 计算微博的情感值
                div_txt = clearTxt(str(post['text']))
                st_cat = 2
                if div_txt:
                    senti_value = sen_value(div_txt)
                    if senti_value > 0.57:
                        st_cat = 1
                    elif senti_value < 0.5:
                        st_cat = 0
                self.st_category.update({node: st_cat})
                nodes.append({
                    # 'attributes': {'modularity_class': 1},
                    'id': node,
                    'category': self.category[node],
                    'itemStyle': '',
                    'label': {'normal': {'show': 'false'}},
                    'name': post['user']['screen_name'],
                    'symbolSize': self.node_size[node],
                    'value': str(post['text']),
                    'x': self.coordinate[node]['x'],
                    'y': self.coordinate[node]['y']
                })
                cur_nodes.append(node)
            link = [str(post['pidstr']), str(post['id'])]
            if link not in cur_links:
                link_id = len(links)
                links.append({
                    'id': link_id,
                    'lineStyle': {'normal': {}},
                    'name': 'null',
                    'source': link[0],
                    'target': link[1]
                })
                cur_links.append(link)



        graph_data = {
            'nodes': nodes,
            'links': links
        }
        return graph_data

    # 计算节点坐标
    def calc_all_node_cor(self):
        nodes_list = [str(self.src_wb['id'])]  # 邻居个数不为零的节点
        while len(nodes_list) != 0:
            node = nodes_list[0]
            nodes_list = self.calc_one_node_cor(node, nodes_list)
            nodes_list.pop(0)

    # 计算节点node邻居节点坐标
    def calc_one_node_cor(self, node, nodes_list):
        num_nbrs = self.network.out_degree(node)  # node的邻居节点数量
        neighbors = self.network.neighbors(node)  # node的邻居节点
        if self.coordinate.get(node):
            node_x = self.coordinate[node]['x']  # node的x坐标
            node_y = self.coordinate[node]['y']  # node的y坐标
        else:
            print('节点{}的坐标不存在'.format(node))
            return nodes_list
        i = 0
        j = 0
        for nbr in neighbors:
            nbr_out = self.network.out_degree(nbr)
            if nbr_out > 0:
                nodes_list.append(nbr)  # nbr节点邻居个数不为零，加入到nodes_list中

            # 计算nbr到父节点的半径r和节点大小
            r = 1.0
            size = 15
            category = self.category[node]
            if num_nbrs < 10:
                r = 0.5 * r
            elif num_nbrs < 400:
                r = r
            else:
                r = 1.5 * r
            if nbr_out > 1 and nbr_out < 10:
                r = 2 * r
                category = self.num_category
                self.num_category += 1
            elif nbr_out >= 10 and nbr_out < 100:
                r = 2.2 * r
                size = size + nbr_out / 10
                category = self.num_category
                self.num_category += 1
            elif nbr_out >= 100:
                r = 2.5 * r
                size = size + 10 + nbr_out / 100
                category = self.num_category
                self.num_category += 1

            # 计算节点坐标
            if num_nbrs == 1:  # 父节点只有一个出边邻居
                # 计算父节点的父节点坐标
                pro_node = next(self.network.predecessors(node))
                pro_node_x = self.coordinate[pro_node]['x']
                pro_node_y = self.coordinate[pro_node]['y']
                nbr_x = node_x + (node_x - pro_node_x) * 0.7
                nbr_y = node_y + (node_y - pro_node_y) * 0.7
                if nbr not in self.coordinate:
                    self.coordinate.update({nbr: {'x': nbr_x, 'y': nbr_y}})
                    self.node_size.update({nbr: size})
                    self.category.update({nbr: category})
            else:
                nbr_x = r * m.cos(i * 2 * m.pi / num_nbrs)
                if nbr_out >= 10:
                    nbr_x = r * m.cos(m.pi / 4)
                    j += 1

                nbr_y = 0
                if i < num_nbrs / 2:
                    nbr_y += m.sqrt(r ** 2 - nbr_x ** 2)
                else:
                    nbr_y -= m.sqrt(r ** 2 - nbr_x ** 2)
                if nbr not in self.coordinate:
                    self.coordinate.update({nbr: {'x': node_x + nbr_x, 'y': node_y + nbr_y}})
                    self.node_size.update({nbr: size})
                    self.category.update({nbr: category})
                i += 1
        return nodes_list


# 分词，去除停用词、英文、符号和数字等
def clearTxt(sentence):
    if sentence != '':
        sentence = sentence.strip()  # 去除文本前后空格
        # 去除文本中的英文和数字
        sentence = re.sub("[a-zA-Z0-9]", "", sentence)
        # 去除文本中的中文符号和英文符号
        sentence = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "", sentence)
        sentence = jieba.lcut(sentence, cut_all=False)
        stopwords = [line.strip() for line in
                     open(data_path_cache + '/stopwords.txt', encoding='gbk').readlines()]
        outstr = ''
        # 去停用词
        for word in sentence:
            if word not in stopwords:
                if word != '\t':
                    outstr += word
                    outstr += " "
        # print(outstr)
        return outstr


def savemodel_snownlp():
    weibo_80k_data = pd.read_csv(data_path + '/weibo_senti_80k.csv', encoding='utf-8')
    col_label = weibo_80k_data.iloc[:, 0].values
    col_content = weibo_80k_data.iloc[:, 1].values
    weibodata = []
    for i in range(len(col_label)):
        weibodata.append([col_label[i], clearTxt(col_content[i])])
    weibodata = pd.DataFrame(weibodata)
    weibodata.columns = ['label', 'comment']
    x = weibodata[['comment']]
    y = weibodata.label
    x = x.comment.apply(clearTxt)
    neg_file = open(data_path_cache + '/neg_file.txt', 'w+', encoding='utf-8')
    pos_file = open(data_path_cache + '/pos_file.txt', 'w+', encoding='utf-8')
    for i in range(len(weibo_80k_data)):
        if y[i] == 0:
            neg_file.write(clearTxt(x[i]) + '\n')
        else:
            pos_file.write(clearTxt(x[i]) + '\n')
    sentiment.train(data_path_cache + '/neg_file.txt',
                    data_path_cache + '/pos_file.txt')  # 训练语料库
    # 保存模型
    sentiment.save(data_path_cache + '/sentiment_snownlp.marshal')


# 求文本的情感倾向值，>0.57则默认为积极，<0.5则默认为消极，0.57与0.5之间可默认为中性
def sen_value(text):
    senti = sn.SnowNLP(text)
    senti_value = round((senti.sentiments), 2)
    return senti_value


# 统计所有文本情感极性，并生成词云图，不考虑时间、省份
def senti_diffusion():
    data = pd.read_csv(data_path + '/weibo.csv', encoding='utf-8')
    content = data.iloc[:, 2].values
    count = [0, 0, 0]  # 统计极性，积极，中性，消极
    for i in range(len(data)):
        data.iloc[i, 2] = clearTxt(str(data.iloc[i, 2]))  # 处理文本
        if data.iloc[i, 2] == '':
            data.iloc[i, 2] = '年月日'  # 即默认为中性态度
        senti_value = sen_value(data.iloc[i, 2])
        if (senti_value >= 0.57):
            count[0] += 1
        elif (senti_value <= 0.5):
            count[2] += 1
        elif (senti_value > 0.5 and senti_value < 0.57):
            count[1] += 1
    print(count)
    # 统计结果：[114092, 10052, 65256]
    # 生成词云
    text = ''
    for j in range(len(data)):
        line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                      str(data.iloc[j, 2]))  # 去除符号
        text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
    backgroud_image = np.array(Image.open(r"static/image/wc_time.jpg"))
    wc = WordCloud(
        background_color='white',  # 设置背景颜色，与图片的背景色相关
        mask=backgroud_image,  # 设置背景图片
        font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
        max_words=300,  # 设置最大显示的字数
        stopwords={'网页', '链接', '博转发'},  # 设置停用词，停用词则不再词云图中表示
        max_font_size=80,  # 设置字体最大值
        random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
        scale=1,  # 设置生成的词云图的大小
    ).generate(text)  # 生成词云
    image_colors = ImageColorGenerator(backgroud_image)
    plt.imshow(wc.recolor(color_func=image_colors))  # 显示词云图
    plt.axis('off')  # 不显示x,y轴坐

    wc.to_file(path.join(data_path_cache + '/weibo_ciyun.png'))
    return count


# 根据时间信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_time():
    time_data = pd.read_excel(data_path_cache + '/weibo_time.xlsx', encoding='utf-8', sheet_name=None)
    sheet_name = time_data.keys()  # 获取当前表格所有sheet名字
    year_count = []  # 保存每一年的积极、消极以及中性文本数目
    for i in sheet_name:
        time_data_y = time_data[i]  # 当前某一年的所有用户数据
        positive = 0
        negative = 0
        neutral = 0
        wc_text = ''  # 当前年份的词云文本

        # 根据年份统计文本情感倾向
        for j in range(len(time_data_y)):
            time_data_y.iloc[j, 2] = clearTxt(str(time_data_y.iloc[j, 2]))  # 由于部分数据只有数字或者符号，需要转成字符型
            if time_data_y.iloc[j, 2] == '':
                time_data_y.iloc[j, 2] = '年月日'  # 即默认为中性态度
            senti_value = sen_value(time_data_y.iloc[j, 2])  # time_data_y.iloc[j,2]表示表格中第三列文本数据
            if (senti_value >= 0.57):
                positive += 1
            elif (senti_value <= 0.5):
                negative += 1
            elif (senti_value > 0.5 and senti_value < 0.57):
                neutral += 1
        year_count.append([positive, neutral, negative])  # 保存每一年的积极、消极以及中性文本数目
        # [[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144], [2916, 420, 1767],
        #  [3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897], [16276, 1868, 13630], [26454, 3719, 23122]]

        # 根据年份文本形成词云图
        for j in range(len(time_data_y)):
            line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                          str(time_data_y.iloc[j, 2]))  # 去除符号
            wc_text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
        backgroud_image = np.array(Image.open(r"static/image/wc_time.jpg"))
        wc_year = WordCloud(
            background_color='white',  # 设置背景颜色，与图片的背景色相关
            mask=backgroud_image,  # 设置背景图片
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
            max_words=300,  # 设置最大显示的字数
            stopwords={'网页', '链接'},  # 设置停用词，停用词则不再词云图中表示
            max_font_size=80,  # 设置字体最大值
            random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=1,  # 设置生成的词云图的大小
        ).generate(wc_text)  # 生成词云
        image_colors = ImageColorGenerator(backgroud_image)
        plt.imshow(wc_year.recolor(color_func=image_colors))  # 显示词云图
        # plt.imshow(wc_year, interpolation='bilinear')
        plt.axis('off')  # 不显示x,y轴坐
        # 按递增顺序保存生成的词云图

        wc_year.to_file(path.join(data_path_cache + '/time_wc', str(i) + '_wc.png'))

    # 生成动态gif文件
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 保存gif文件
        imageio.mimsave(gif_name, frames, 'gif', duration=0.3)
        return

    def find_all_gif():

        png_filenames = glob.glob(data_path_cache + '/result/time_wc_chuo/*')  # 加入图片位置，绝对路径
        buf = []
        for png_file in png_filenames:
            buf.append(png_file)
        return buf

    # 为每张图片生成时间戳
    png_list = os.listdir(data_path_cache + '/time_wc')
    font = ImageFont.truetype("C:\Windows\Fonts\STZHONGS.TTF", 40)
    j = 0
    dir_list = os.listdir(data_path_cache + '/time_wc/')
    dir = []

    for i in dir_list:
        dir.append(re.sub('_wc', '', str(os.path.splitext(i)[0])))
    for png in png_list:
        name = png
        imageFile = data_path_cache + '/time_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                  dir[j] + '年',  # 设置内容
                  (0, 0, 0),  # 设置颜色
                  font=font)  # "设置字体
        draw = ImageDraw.Draw(im)
        # 存储图片
        im.save(data_path_cache + '/result/time_wc_chuo/' + name)
        j += 1
    buff = find_all_gif()
    time_wc = create_gif(buff, data_path_cache + '/result/time_wc.gif')  # 生成时间词云动态gif图
    return year_count, time_wc


# 根据地域信息判断所有用户的情感倾向，生成词云图
def senti_diffusion_position():
    position_data = pd.read_excel(data_path_cache + '/weibo_position.xlsx', encoding='utf-8', sheet_name=None)
    sheet_name = position_data.keys()  # 获取当前表格所有sheet名字
    # print(sheet_name)
    position_count = []  # 保存每个省份的积极、消极以及中性文本数目
    for i in sheet_name:
        position_data_y = position_data[i]  # 当前某省份的所有用户数据
        positive = 0
        negative = 0
        neutral = 0
        wc_text = ''  # 当前省份的词云文本

        # 根据省份统计文本情感倾向
        for j in range(len(position_data_y)):
            position_data_y.iloc[j, 2] = clearTxt(str(position_data_y.iloc[j, 2]))  # 由于部分数据只有数字或者符号，需要转成字符型
            if position_data_y.iloc[j, 2] == '':
                position_data_y.iloc[j, 2] = '年月日'  # 即默认为中性态度
            senti_value = sen_value(position_data_y.iloc[j, 2])  # position_data_y.iloc[j,2]表示表格中第三列文本数据
            if (senti_value >= 0.57):
                positive += 1
            elif (senti_value <= 0.5):
                negative += 1
            elif (senti_value > 0.5 and senti_value < 0.57):
                neutral += 1
        position_count.append([i, positive, neutral, negative])  # 保存每个省份的积极、消极以及中性文本数目
        # [['黑龙', 1428, 246, 1270], ['北京', 5505, 677, 4699], ['辽宁', 8121, 1290, 6342], ['内蒙', 1149, 165, 1121],
        #  ['香港', 577, 67, 431], ['天津', 1107, 138, 922], ['云南', 1240, 129, 1007], ['湖南', 2125, 194, 1533],
        #  ['河南', 2083, 230, 1621], ['山东', 2873, 250, 2488], ['西藏', 370, 51, 438], ['广西', 1077, 131, 962],
        #  ['山西', 1356, 127, 1359], ['台湾', 818, 56, 444], ['新疆', 781, 86, 666], ['江西', 983, 140, 1218], ['吉林', 2066, 273, 1208],
        #  ['河北', 1928, 225, 1507], ['四川', 2518, 299, 2027], ['甘肃', 1001, 131, 695], ['福建', 2465, 344, 1761],
        #  ['广东', 6847, 912, 5948], ['安徽', 1425, 132, 962], ['浙江', 3947, 406, 3044], ['上海', 3677, 384, 2987],
        #  ['陕西', 1744, 160, 1163], ['澳门', 474, 51, 484], ['海外', 4319, 628, 2467], ['江苏', 4027, 464, 3104],
        #  ['湖北', 1712, 193, 1362], ['海南', 697, 136, 663], ['贵州', 1025, 106, 806], ['重庆', 1275, 167, 987],
        #  ['其他', 8699, 966, 6681], ['青海', 637, 57, 476], ['宁夏', 397, 24, 275]]

        # 根据省份文本形成词云图
        for j in range(len(position_data_y)):
            line = re.sub("[\s+\.\!\/_,$%^*(+]\"\']+|[+——！，:\[\]。!：？?～”,、\.\/~@#￥%……&*【】 （）]+", "",
                          str(position_data_y.iloc[j, 2]))  # 去除符号
            wc_text += ' '.join(jieba.cut(line, cut_all=True))
        # 设置词云格式
        backgroud_Image = np.array(Image.open("static/image/wc_time.jpg"))
        wc_position = WordCloud(
            background_color='white',  # 设置背景颜色，与图片的背景色相关
            mask=backgroud_Image,  # 设置背景图片
            font_path='C:\Windows\Fonts\STZHONGS.TTF',  # 显示中文，可以更换字体
            max_words=300,  # 设置最大显示的字数
            stopwords={'网页', '链接'},  # 设置停用词，停用词则不再词云图中表示
            max_font_size=80,  # 设置字体最大值
            random_state=5,  # 设置有多少种随机生成状态，即有多少种配色方案
            scale=1  # 设置生成的词云图的大小
        )
        wc_position.generate(wc_text)  # 生成词云
        image_colors = ImageColorGenerator(backgroud_Image)
        plt.imshow(wc_position.recolor(color_func=image_colors))  # 显示词云图
        plt.axis('off')  # 不显示x,y轴坐标
        # 按递增顺序保存生成的词云图
        wc_position.to_file(path.join(data_path_cache + '/position_wc', str(i) + '_wc.png'))

    # 生成动态gif文件
    def create_gif(image_list, gif_name):
        frames = []
        for image_name in image_list:
            frames.append(imageio.imread(image_name))
        # 保存gif文件
        imageio.mimsave(gif_name, frames, 'gif', duration=0.3)
        return

    def find_all_gif():
        png_filenames = glob.glob(data_path_cache + '/result/position_wc_chuo/*')  # 加入图片位置，绝对路径
        buf = []
        for png_file in png_filenames:
            buf.append(png_file)
        return buf

    # 为每张图片生成地域省份戳
    png_list = os.listdir(data_path_cache + '/position_wc')
    font = ImageFont.truetype("C:\Windows\Fonts\STZHONGS.TTF", 40)
    j = 0
    dir_list = os.listdir(data_path_cache + '/position_wc/')
    dir = []
    for i in dir_list:
        position_n = os.path.splitext(i)[0]
        if position_n == '黑龙_wc':
            position_n = '黑龙江_wc'
        if position_n == '内蒙_wc':
            position_n = '内蒙古_wc'
        dir.append(re.sub('_wc', '', str(position_n)))
    for png in png_list:
        name = png
        imageFile = data_path_cache + '/position_wc/' + name
        im = Image.open(imageFile)
        draw = ImageDraw.Draw(im)
        draw.text((0, 0),  # 设置字体位置
                  dir[j],  # 设置内容
                  (0, 0, 0),  # 设置颜色
                  font=font)  # "设置字体
        draw = ImageDraw.Draw(im)
        # 存储图片
        im.save(data_path_cache + '/result/position_wc_chuo/' + name)
        j += 1
    buff = find_all_gif()
    position_wc = create_gif(buff, data_path_cache + '/result/position_wc.gif')  # 生成时间词云动态gif图
    return position_count, position_wc

weibo, num_wb = load_wb_data('static/data/weibo/weibo.csv')
user, num_ur = load_ur_data('static/data/weibo/userInfo.csv')
repost = Repost('static/data/weibo/repost.csv')


# 谣言可视化部分支撑
def SD(iteration,percentage,method,index):
    count = 0  # 准确定位到源的次数
    errorDistance = [0 for index in range(5)]  # 存放每一跳的误差比例
    all_iteration_dis = []  # [[真实的源，预测的源，第一次迭代误差距离]，第二次迭代预测源与真实源的误差距离，.....,误差列表，所有迭代中的准确率]
    distance = 0
    mean_error_distance = 0  # 平均误差距离
    shortestPath = []  # [[[6(reverse_source), 35(reverse_target), 98.0(边编号）]], [[2, 5, 33.0], [5, 6, 75.0], [6, 35, 98.0]], 记录每个最短路径
    shortestPath1 = []  # 最短路径列表[[a,b,c],[c,f,g,e]....]
    active_records1 = []
    edge_records1 = []
    ObserverNodeList1 = []
    for i in range(int(iteration)):
        if(index==1):
            candidateCommunity, candidateCommunityObserveInfectedNode, ALLCandidatSourceNode, AllCandidateObserveNode, relSource, CommunitiesList, \
            SourceNodeInCom, ObserverNodeList, active_records, edge_records, new_node_neigbors_dic = sd.SI_diffusion(
                percentage,method)
            preSource, maxValue = sd.GM(ALLCandidatSourceNode, AllCandidateObserveNode, new_node_neigbors_dic)
        else:
            candidateCommunity, candidateCommunityObserveInfectedNode, ALLCandidatSourceNode, AllCandidateObserveNode, relSource, CommunitiesList, \
            SourceNodeInCom, ObserverNodeList, active_records, edge_records = GE.SI_diffusion(percentage,method)
            preSource, maxValue = GE.GM(ALLCandidatSourceNode, AllCandidateObserveNode)
        if (preSource == relSource):
            count += 1
            errorDistance[0] += 1
            distance = 0
        else:
            distance = nx.shortest_path_length(sd.G, preSource, relSource)
            errorDistance[distance] += 1
        all_iteration_dis.append([relSource, preSource, distance])
        if i == 0:
            ObserverNodeList1 = ObserverNodeList
            active_records1 = json.dumps(active_records)
            edge_records1 = json.dumps(edge_records)
            for i in ObserverNodeList1:
                shortestPath1.append(nx.shortest_path(sd.G, i, relSource))
            for paths in shortestPath1:
                shortestPath.append([])
                for i in range(len(paths) - 1):
                    shortestPath[len(shortestPath) - 1].append(
                        [paths[i], paths[i + 1], sd.edgeNum[paths[i]][paths[i + 1]]])
    for j in range(len(errorDistance)):
        mean_error_distance += errorDistance[j] * j
        errorDistance[j] = round(errorDistance[j] / int(iteration), 2)  # 误差在各跳数的比例
    mean_error_distance = round(mean_error_distance / int(iteration), 2)  # 平均误差距离
    errorDistance = [e for e in errorDistance if e > 0]
    all_iteration_dis.extend(
        [errorDistance, round(count / int(iteration), 2), mean_error_distance])  # 误差列表，定位准确率，平均误差距离
    return ObserverNodeList1,active_records1,edge_records1,shortestPath,all_iteration_dis


def inputJudge(percentage,iteration,method,err):
    if percentage != "":
        if not re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$').match(percentage):
            err = "输入不合法，请输入0.05-0.9之间的数字"
        elif float(percentage) < 0.05 or float(percentage) > 0.9:
            err = "输入错误，请输入0.05-0.9之间的数字"
    elif percentage == "":
        err = "输入为空，请输入观测比例"
    if iteration == "":
        err = "输入为空，请输入迭代次数"
    else:
        if iteration.isdigit() == False:
            err = "输入不合法，请输入一个整数"
        elif int(iteration) < 1 or int(iteration) > 10000:
            err = "迭代次数最少为1，请输入大于1的整数"
    return err

# 主框架
async_mode = None
thread = None
app = Flask(__name__)
app.secret_key = 'lisenzzz'
socketio = SocketIO(app, async_mode=async_mode)
connection = sqlite3.connect("logindata.db")
cur = connection.cursor()
# cur.execute("delete from udata where user = 'Jagger'")

cur.execute('CREATE TABLE IF NOT EXISTS udata (user varchar(128) PRIMARY KEY, number varchar(11), '
            'mail varchar(32), unit varchar(32), password varchar(128))')
app.config['MAIL_SERVER'] = 'smtp.qq.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = '3517717683@qq.com'
app.config['MAIL_PASSWORD'] = 'psbpzckhcmqncibf'
mail = Mail(app)

@app.route('/checkUser', methods=["POST"])
def checkUser():
    if request.method == 'POST':
        sqlite3.connect('logindata.db')
        requestArgs = request.values
        user = requestArgs.get('user')
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()
        if result is None:
            return jsonify({'isExist': False})
        elif result is not None:
            return jsonify({'isExist': True})


@app.route('/forget', methods=["GET", "POST"])
def forget():
    if request.method == 'GET':
        return render_template('forget.html')
    elif request.method == 'POST':
        requestArgs = request.values
        new = requestArgs.get('password')
        user = requestArgs.get('user')
        cur.execute("update udata set password='" + new + "' where user='" + user + "'")
        connection.commit()
        return jsonify({'isSuccess': 1})


@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    elif request.method == 'POST':
        requestArgs = request.values
        user = requestArgs.get('user')
        password = requestArgs.get('password')
        number = requestArgs.get('number')
        unit = requestArgs.get('unit')
        mail = requestArgs.get('mail')
        str = "'" + user + "'" + ",'" + number + "'," + "'" + mail + "'" + "," \
              + "'" + unit + "'" + "," + "'" + password + "'"
        cur.execute('insert into udata (user,number,mail,unit,password) values (' + str + ")")
        connection.commit()
        return jsonify({'isSuccess': 1})


@app.route('/send', methods=["POST"])
def send():
    requestArgs = request.values
    dirMail = requestArgs.get('mail')
    user = requestArgs.get('user')
    # print(user)

    if user is not None:
        cur.execute("select * from udata where user = " + "'" + user + "'")
        # print("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()
        # print("mail" + result[2])
        if result != None:
            if result[2] != dirMail:
                return jsonify({'ischecked': 0})
    verificationList = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                        'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 's', 't', 'x', 'y', 'z', 'A', 'B', 'C', 'D',
                        'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'S', 'T', 'X', 'Y', 'Z']
    veriCode = ''
    for i in range(4):
        veriCode += verificationList[random.randint(0, len(verificationList) - 1)]
    msg = Message("可视化平台验证码", sender="3517717683@qq.com", recipients=[dirMail])
    msg.body = veriCode
    mail.send(msg)
    return jsonify({'code': veriCode, 'ischecked': 1})


@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'GET':
        sqlite3.connect('logindata.db')
        return render_template('login.html')
    elif request.method == 'POST':
        requestArgs = request.values
        user = requestArgs.get('user')
        password = requestArgs.get('password')
        # print("password" + password)
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()  # 没找到为None, 否则返回对应的元组
        cur.execute("select * from udata where password = " + "'" + password + "'")
        # p = cur.fetchone()  # 返回的是三元组，p[0]是需要的值
        # for i in range(5):
        #     print(result[i] + "result" )
        check = {'userInfo': -1, 'passwordInfo': -1}
        if result is None:
            check['userInfo'] = -1
        elif result is not None:
            check['userInfo'] = 0
            if password == result[4]:
                check['passwordInfo'] = 1
            elif password != result[4]:
                check['passwordInfo'] = 0
        check = json.dumps(check)
        return jsonify({'check': check})


@app.route('/fun', methods=["POST"])
def fun():
    requestArgs = request.values
    user = requestArgs.get('userName')
    cur.execute("select * from udata where user = " + "'" + user + "'")
    result = cur.fetchone()
    return jsonify({'user': result})

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
    if request.method == 'POST':
        sqlite3.connect('logindata.db')
        requestArgs = request.values
        user = requestArgs.get('user')
        # print(user)
        # if user is None:
        #     return jsonify({'isSuccess': 0})
        cur.execute("select * from udata where user = " + "'" + user + "'")
        result = cur.fetchone()
        # print(type(result[2]), type(result[3]),type(result[1]))
        if result is None:
            return jsonify({'isSuccess': 0})
        elif result[0] == user:
            userInfo['user'] = result[0]
            userInfo['phoneNumber'] = result[1]
            userInfo['mail'] = result[2]
            userInfo['unit'] = result[3]
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
    data = {'graph_data': repost.graph_data,
            'node_num': repost.num_reposts,
            'num_category': repost.num_category,
            'category': repost.category,
            'st_category': repost.st_category}  # 所有要传给前端的数据

    # 统计不同时间的转发微博
    times = repost.reposts.sort_values(by='created_at')['created_at'].drop_duplicates().to_list()  # 所有时间集合
    rp_records = []  # 记录每个时间段内的微博编号
    for t in times:
        posts_id = repost.reposts[repost.reposts['created_at'] == t]['id'].to_list()  # t时刻转发的微博ID
        # 将微博id转换为微博编号
        post_indexs = []
        for post_id in posts_id:
            post_indexs.append(repost.post_indexs[str(post_id)])
        rp_records.append({'time': t, 'post_indexs': post_indexs})
    data['rp_records'] = rp_records

    weiboRepost = repost.reposts
    num_repost = repost.num_reposts
    users = weiboRepost.loc[:, 'user']
    # 统计转发微博的用户的性别比例
    f_count = 0
    m_count = 0
    for i in range(1, num_repost):  # 第一条微博是原微博，不计入统计
        if users[i]['gender'] == 'f':
            f_count += 1
        elif users[i]['gender'] == 'm':
            m_count += 1
    gender_rate = {"female": int(f_count) / num_repost}
    gender_rate.update({"male": int(m_count) / num_repost})
    # 传给前端的转发微博用户性别比例数据
    gender = {
        "num_wb": int(num_wb),
        "gender_rate": gender_rate
    }
    data['gender'] = gender

    # 统计转发微博用户认证比例
    verified_count = 0
    unVerified_count = 0
    for i in range(1, num_repost):
        if users[i]['verified']:
            verified_count += 1
        elif not users[i]['verified']:
            unVerified_count += 1
    verified_rate = {"认证": int(verified_count) / num_repost}
    verified_rate.update({"非认证": int(unVerified_count) / num_repost})
    verified = {
        "verified_rate": verified_rate
    }
    data['verified'] = verified

    # 转发层级
    startP = repost.src_wb  # 源微博
    graph = repost.network  # 关系图
    nodes_list = list(graph.neighbors(str(startP[2])))  # 源微博的邻居节点，也就是一级转发
    level_num = 0  # 级数
    level_dic = []  # 级数-数量
    visited = [str(startP[2])]  # 已经算过的节点
    level_rate = []  # 每一级微博数量占所有微博的比例

    while len(nodes_list) != 0:  # 类似于广度遍历
        length = len(nodes_list)
        level_num += 1
        if level_num == 1:
            level_dic = {'1': str(length)}
            level_rate = {'1': length / num_repost}
        else:
            level_dic.update({str(level_num): str(length)})
            level_rate.update({str(level_num): length / num_repost})
        new_neigh_list = []  # 下一级节点
        for i in range(length):
            now_node_list = list(graph.neighbors(nodes_list[i]))  # 当前节点的邻居节点
            for j in range(len(now_node_list)):  # 遍历新的邻居节点
                if visited.count(now_node_list[j]):  # 已经遍历过该节点
                    break
                visited.append(now_node_list[j])  # 加入到已遍历节点
                new_neigh_list.append(now_node_list[j])
        nodes_list = new_neigh_list
    level = {
        "level_dic": level_dic,
        "level_rate": level_rate,
    }
    data['level'] = level

    # 转发量-时间
    time_count = weiboRepost.loc[:, 'created_at'].value_counts()
    startP = weiboRepost.loc[0]
    startTime = startP['created_at'].split("-")
    d1 = datetime.datetime(int(startTime[0]), int(startTime[1]), int(startTime[2]))
    max_day = 0
    for i in range(len(time_count)):
        nowTime = time_count.index[i].split("-")
        d2 = datetime.datetime(int(nowTime[0]), int(nowTime[1]), int(nowTime[2]))
        interval = d2 - d1  # 两日期差距
        if interval.days > max_day:
            max_day = interval.days
        if i == 0:
            time_repost1 = {interval.days: time_count[i]}
        else:
            time_repost1.update({interval.days: time_count[i]})
    for i in range(max_day + 1):
        if i in time_repost1.keys():
            if i == 0:
                time_repost = {str(i): (time_repost1[i] - 1) / 1.0}
            else:
                time_repost.update({str(i): time_repost1[i] / 1.0})
        else:
            if i == 0:
                time_repost = {str(i): 0 / 1.0}
            else:
                time_repost.update({str(i): 0 / 1.0})
    time = {
        "time_repost": time_repost,
    }
    data['time'] = time

    data_json = json.dumps(data)
    return render_template('AnalysisOfWeiboForwardingStructure.html', data_json=data_json)

# 单条微博情感分析，后端服务
@app.route('/EmotionalAnalysisOfSingleWeibo', methods=['GET', 'POST'])
def EmotionalAnalysisOfSingleWeibo():
    global senti_value
    global content_err
    if request.method == 'POST':
        content = request.form.get('content_value')
        if content != '':
            content_err = 1
            senti_value = sen_value(clearTxt(content))
        else:
            senti_value = 0
            content_err = 0
        senti_value = json.dumps(senti_value)
        content_err = json.dumps(content_err)
        return render_template('EmotionalAnalysisOfSingleWeibo.html', senti_value=senti_value, content_err=content_err)
    else:
        return render_template('EmotionalAnalysisOfSingleWeibo.html')

# 微博情感整体分析，后端服务
@app.route('/OverallAnalysisOfWeiboSentiment', methods=['GET', 'POST'])
def OverallAnalysisOfWeiboSentiment():
    with open(data_path_cache + "/result/user_position_count.txt", "r",
              encoding='utf-8') as user_position_count:
        data = user_position_count.readlines()
        user_position = []  # 用户省份地址
        user_position_c = []  # 省份用户人数
        user_text_count = [114092, 10052, 65256]  # 用户微博极性统计，根据senti_diffusion()获得
        for i in range(len(data)):
            user_position.append(data[i][0:2])
            user_position_c.append(int(data[i][3:]))

    def getresult(path):
        user_senticount = pd.read_table(path)
        return user_senticount

    if request.method == 'GET':
        user_position = json.dumps(user_position)
        user_position_c = json.dumps(user_position_c)
        user_text_count = json.dumps(user_text_count)
        user_ciyun_path = data_path_cache + '/weibo_ciyun.png'
        return render_template('OverallAnalysisOfWeiboSentiment.html',user_position_c=user_position_c,
                               user_position=user_position, user_text_count=user_text_count,
                               user_ciyun_path=user_ciyun_path)

# 微博情感详细分析，后端服务
@app.route('/DetailedAnalysisOfWeiboSentiment', methods=['GET', 'POST'])
def DetailedAnalysisOfWeiboSentiment():
    # 按照时间用户情感极性分布
    user_time_senticount = [[3, 5, 13], [763, 105, 576], [876, 109, 805], [1453, 140, 1095], [5627, 497, 3144],
                            [2916, 420, 1767], [3398, 502, 3131], [7328, 940, 6149], [15219, 1579, 10897],
                            [16276, 1868, 13630], [26454, 3719, 23122]]
    user_position_senticount = [['黑龙江', 1428, 246, 1270], ['北京', 5505, 677, 4699], ['辽宁', 8121, 1290, 6342],
                                ['内蒙古', 1149, 165, 1121], ['香港', 577, 67, 431], ['天津', 1107, 138, 922],
                                ['云南', 1240, 129, 1007], ['湖南', 2125, 194, 1533], ['河南', 2083, 230, 1621],
                                ['山东', 2873, 250, 2488], ['西藏', 370, 51, 438], ['广西', 1077, 131, 962],
                                ['山西', 1356, 127, 1359], ['台湾', 818, 56, 444], ['新疆', 781, 86, 666],
                                ['江西', 983, 140, 1218], ['吉林', 2066, 273, 1208], ['河北', 1928, 225, 1507],
                                ['四川', 2518, 299, 2027], ['甘肃', 1001, 131, 695], ['福建', 2465, 344, 1761],
                                ['广东', 6847, 912, 5948], ['安徽', 1425, 132, 962], ['浙江', 3947, 406, 3044],
                                ['上海', 3677, 384, 2987], ['陕西', 1744, 160, 1163], ['澳门', 474, 51, 484],
                                ['海外', 4319, 628, 2467], ['江苏', 4027, 464, 3104], ['湖北', 1712, 193, 1362],
                                ['海南', 697, 136, 663], ['贵州', 1025, 106, 806], ['重庆', 1275, 167, 987],
                                ['其他', 8699, 966, 6681], ['青海', 637, 57, 476], ['宁夏', 397, 24, 275]]
    user_position_senticount = json.dumps(user_position_senticount)
    user_time_senticount = json.dumps(user_time_senticount)
    user_timeciyun_path = data_path_cache + '/result/time_wc_chuo'  # 时间词云文件夹路径
    user_positionciyun_path = data_path_cache + '/result/position_wc_chuo'  # 省份词云文件夹路径
    user_timeciyun_path = json.dumps(user_timeciyun_path)
    user_positionciyun_path = json.dumps(user_positionciyun_path)
    if request.method == 'GET':
        return render_template('DetailedAnalysisOfWeiboSentiment.html',user_time_senticount=user_time_senticount,
                               user_position_senticount=user_position_senticount,
                               user_timeciyun_path=user_timeciyun_path,
                               user_positionciyun_path=user_positionciyun_path)

# 介绍SI模型，后端服务
@app.route('/introduceSIModel')
def introduceSIModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceSIModel.html')

# SI模型展示，后端服务
@app.route('/DemonstrationOfSIModel', methods=["GET", "POST"])
def DemonstrationOfSIModel():
    graph_data1 = json.loads(si.graph_data)  # 将json数据转化为字典的形式
    nodeset, num_node = si.CalculateNodesnum()
    Susceptible = nodeset  # 易感者集合
    Infected = []  # 感染者集合
    edge = []  # 边
    times = []  # 保存经历了多少时间步

    sus_node = []  # 保存每一时间步易感者节点集合
    inf_node = []  # 保存每一时间步感染者节点集合
    edges_record = []  # 保存边
    sus_node_num = []  # 保存每一时间步易感节点数量
    inf_node_num = []  # 保存每一时间步感染节点数量

    # 随机生成一个感染源节点，并从易感节点移除加入感染节点集

    inf = random.randint(1, num_node)
    Infected.append(inf)
    Susceptible.remove(inf)
    # 打印
    # print("Inf_node:", inf)
    # print("Inf_Susceptible:", Susceptible, len(Susceptible))
    # print("Inf_Infected:", Infected, len(Infected))
    t = 0
    err = "false"
    # SI模拟
    if request.method == "GET":
        sus_node = json.dumps([])
        inf_node = json.dumps([])
        edges_record = json.dumps([])
        err = "true"
        rateSI = 0
        return render_template('DemonstrationOfSIModel.html', graph_data=si.graph_data, susceptible_nodes=sus_node,
                               sus_node_num=sus_node_num,
                               infected_nodes=inf_node, edges_record=edges_record, inf_node_num=inf_node_num,
                               times=times, err=err, rateSI=rateSI, method_type=1)
    if request.method == "POST":
        rateSI = request.form.get('rateSI')
        if rateSI != "":
            if not re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$').match(rateSI):
                err = "输入不合法，请输入0-1区间的值"
            elif float(rateSI) < 0 or float(rateSI) > 1:
                err = "输入错误，请输入0-1区间的值"
        elif rateSI == "":
            err = "请输入感染率"
    if err != "false":
        return render_template('DemonstrationOfSIModel.html', graph_data=si.graph_data, susceptible_nodes=json.dumps([]),
                               infected_nodes=json.dumps([]), edges_record=json.dumps([]),
                               sus_node_num=sus_node_num, inf_node_num=inf_node_num,
                               times=times, err=err, rateSI=0, method_type=1)
    while len(Susceptible) != 0:
        sus_node.append(copy.deepcopy(Susceptible))
        inf_node.append(copy.deepcopy(Infected))
        edges_record.append(copy.deepcopy(edge))

        sus_node_num.append(len(Susceptible))
        inf_node_num.append(len(Infected))
        t = t + 1
        # print(type(rateSI))
        Susceptible, Infected, edge = si.SIsimulation(float(rateSI), Susceptible, Infected)
        # print(t, ":Susceptible:", Susceptible, len(Susceptible))
        # print(t, ":Infected:", Infected, len(Infected))
        # print(t, "edge", edge, len(edge))
    sus_node.append(copy.deepcopy(Susceptible))
    inf_node.append(copy.deepcopy(Infected))
    edges_record.append(copy.deepcopy(edge))

    sus_node_num.append(len(Susceptible))
    inf_node_num.append(len(Infected))

    for i in range(1, t + 2):
        times.append(i)
    # print("times:",times)
    # print("edge:", edges_record)
    sus_node1 = []
    sus_node1.append(sus_node[0])
    for i in range(1, len(sus_node)):
        sus_node1.append([])
        for j in range(len(sus_node[i])):
            if (sus_node[i][j] not in sus_node[i - 1]):
                sus_node1[i].append(sus_node[i][j])
    # print(sus_node1, len(sus_node1))

    inf_node1 = []
    inf_node1.append(inf_node[0])
    for i in range(1, len(inf_node)):
        inf_node1.append([])
        for j in range(len(inf_node[i])):
            if (inf_node[i][j] not in inf_node[i - 1]):
                inf_node1[i].append(inf_node[i][j])
    # print(inf_node1, len(inf_node1))

    sus_node = json.dumps(sus_node1)
    inf_node = json.dumps(inf_node1)
    edges_record = json.dumps(edges_record)

    # print("inf_node_num", inf_node_num)

    graph_data1 = json.dumps(graph_data1)  # 将数据转化为json格式
    return render_template('DemonstrationOfSIModel.html', graph_data=graph_data1, susceptible_nodes=sus_node, infected_nodes=inf_node,
                           edges_record=edges_record, sus_node_num=sus_node_num, inf_node_num=inf_node_num,
                           times=times, err=err, rateSI=rateSI, method_type=1)

# 介绍SIR模型，后端服务
@app.route('/introduceSIRModel')
def introduceSIRModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceSIRModel.html')

# SIR模型展示，后端服务
@app.route('/DemonstrationOfSIRModel', methods=["GET", "POST"])
def DemonstrationOfSIRModel():
    graph_data1 = json.loads(sir.graph_data)  # 将json数据转化为字典的形式
    nodeset, num_node = sir.CalculateNodesnum()
    Susceptible = nodeset  # 易感者集合
    Infected = []  # 感染者集合
    Resistant = []  # 恢复者集合
    edge = []  # 边
    times = []  # 保存经历了多少时间步

    sus_node = []  # 保存每一时间步易感节点集合
    inf_node = []  # 保存每一时间步感染节点集合
    res_node = []  # 保存每一时间步恢复节点集合
    edges_record = []  # 保存边

    sus_node_num = []  # 保存每一时间步易感节点数量
    inf_node_num = []  # 保存每一时间步感染节点数量
    res_node_num = []  # 保存每一时间步恢复节点的数量

    # 随机生成一个感染源节点，并从易感节点移除加入感染节点集
    inf = random.randint(1, num_node)
    Infected.append(inf)
    Susceptible.remove(inf)

    # print("Inf_node:", inf)
    # print("Inf_Susceptible:", Susceptible, len(Susceptible))
    # print("Inf_Infected:", Infected, len(Infected))
    # print("Inf_Resistant:", Resistant, len(Resistant))

    t = 0
    err = "false"  # 返回错误信息
    # SIR模拟
    if request.method == "GET":
        sus_node = json.dumps([])
        inf_node = json.dumps([])
        res_node = json.dumps([])
        edges_record = json.dumps([])
        err = "true"
        rateSI = 0
        rateIR = 0

        return render_template('DemonstrationOfSIRModel.html', graph_data=sir.graph_data, susceptible_nodes=sus_node,
                               infected_nodes=inf_node, resistant_nodes=res_node, edges_record=edges_record,
                               sus_node_num=sus_node_num, inf_node_num=inf_node_num, res_node_num=res_node_num,
                               times=times, err=err, rateIR=rateIR, rateSI=rateSI, method_type=1)
    if request.method == "POST":
        rateSI = request.form.get('rateSI')
        rateIR = request.form.get('rateIR')
        if rateSI != "":
            if not re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$').match(rateSI):
                err = "输入不合法，请输入0-1区间的值"
            elif float(rateSI) < 0 or float(rateSI) > 1:
                err = "输入错误，请输入0-1区间的值"
        elif rateSI == "":
            err = "请输入感染率"
        if rateIR != "":
            if not re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$').match(rateIR):
                err = "输入不合法，请输入0-1区间的值"
            elif float(rateIR) < 0.05 or float(rateIR) > 0.9:
                err = "输入错误，请输入0-1区间的值"
        elif rateIR == "":
            err = "请输入恢复率"

        if (rateIR == "" and rateSI == ""):
            err = "请输入感染率和恢复率"
    if err != "false":
        return render_template('DemonstrationOfSIRModel.html', graph_data=sir.graph_data, susceptible_nodes=json.dumps([]),
                               infected_nodes=json.dumps([]), resistant_nodes=json.dumps([]),
                               edges_record=json.dumps([]),
                               sus_node_num=sus_node_num, inf_node_num=inf_node_num, res_node_num=res_node_num,
                               times=times, err=err, rateIR=0, rateSI=0, method_type=1)

    while len(Infected) != 0:
        sus_node.append(copy.deepcopy(Susceptible))
        inf_node.append(copy.deepcopy(Infected))
        res_node.append(copy.deepcopy(Resistant))
        edges_record.append(copy.deepcopy(edge))

        sus_node_num.append(len(Susceptible))
        inf_node_num.append(len(Infected))
        res_node_num.append(len(Resistant))
        t = t + 1
        # print(type(rateSI))
        # print(type(rateIR))
        Susceptible, Infected, Resistant, edge = sir.SIRsimulation(float(rateSI), float(rateIR), Susceptible, Infected,
                                                                   Resistant)
        # print(t, ":Susceptible:", Susceptible, len(Susceptible))
        # print(t, ":Infected:", Infected, len(Infected))
        # print(t, ":Resistant:", Resistant, len(Resistant))
        # print(t, "edge", edge, len(edge))

    sus_node.append(copy.deepcopy(Susceptible))
    inf_node.append(copy.deepcopy(Infected))
    res_node.append(copy.deepcopy(Resistant))
    edges_record.append(copy.deepcopy(edge))

    sus_node_num.append(len(Susceptible))
    inf_node_num.append(len(Infected))
    res_node_num.append(len(Resistant))

    for i in range(1, t + 2):
        times.append(i)
    # print("times:",times)
    #
    # print("edge:", edges_record)
    sus_node1 = []
    sus_node1.append(sus_node[0])
    for i in range(1, len(sus_node)):
        sus_node1.append([])
        for j in range(len(sus_node[i])):
            if (sus_node[i][j] not in sus_node[i - 1]):
                sus_node1[i].append(sus_node[i][j])
    # print(sus_node1, len(sus_node1))

    inf_node1 = []
    inf_node1.append(inf_node[0])
    for i in range(1, len(inf_node)):
        inf_node1.append([])
        for j in range(len(inf_node[i])):
            if (inf_node[i][j] not in inf_node[i - 1]):
                inf_node1[i].append(inf_node[i][j])
    # print(inf_node1, len(inf_node1))

    res_node1 = []
    res_node1.append(res_node[0])
    for i in range(1, len(res_node)):
        res_node1.append([])
        for j in range(len(res_node[i])):
            if (res_node[i][j] not in res_node[i - 1]):
                res_node1[i].append(res_node[i][j])
    # print(res_node1, len(res_node1))

    sus_node = json.dumps(sus_node1)
    inf_node = json.dumps(inf_node1)
    res_node = json.dumps(res_node1)
    edges_record = json.dumps(edges_record)

    # print("inf_node_num", inf_node_num)
    # print("res_node_num",res_node_num)

    graph_data1 = json.dumps(graph_data1)  # 将数据转化为json格式
    return render_template('DemonstrationOfSIRModel.html', graph_data=graph_data1, susceptible_nodes=sus_node,
                           infected_nodes=inf_node, resistant_nodes=res_node, edges_record=edges_record,
                           sus_node_num=sus_node_num, inf_node_num=inf_node_num, res_node_num=res_node_num,
                           times=times, err=err, rateIR=rateIR, rateSI=rateSI, method_type=1)


# 介绍谣言溯源模型，后端服务
@app.route('/introduceRumorTraceabilityModel')
def introduceRumorTraceabilityModel():
    '''
    TODO
    :return:
    '''
    return render_template('introduceRumorTraceabilityModel.html')

# 谣言溯源模型展示，后端服务
@app.route('/DemonstrationOfRumorTraceabilityModel', methods=["GET", "POST"])
def DemonstrationOfRumorTraceabilityModel():
    node_in_Community = sd.node_in_Community  # 每个节点所在的分区
    err = "false"  # 返回错误信息
    if request.method == "GET":
        # 初始化权重矩阵
        err = "true"
        active_records = json.dumps([])
        ObserverNodeList = []
        edge_records = json.dumps([])
        return render_template('DemonstrationOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                               ObserverNodeList=ObserverNodeList, active_records=active_records,
                               edge_records=edge_records,
                               shortestPath=[], err=err, node_in_Community=node_in_Community)
    else:  # request：请求对象，获取请求方式数据
        percentage = request.form.get('percentage')  # 观测点数量
        iteration = request.form.get('iteration')  # 迭代次数
        method = request.form.get('method')  # 选择观测点方法
        err=inputJudge(percentage, iteration, method, err)
    if err != "false":
        return render_template('DemonstrationOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                               ObserverNodeList=[], active_records=json.dumps([]), edge_records=json.dumps([]),
                               shortestPath=[], err=err, node_in_Community=node_in_Community)
    ObserverNodeList, active_records, edge_records, shortestPath, all_iteration_dis = SD(iteration,
                                                                                              float(percentage),
                                                                                            int(method), 1)
    return render_template('DemonstrationOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                           ObserverNodeList=ObserverNodeList,
                           active_records=active_records, edge_records=edge_records,
                           shortestPath=shortestPath, err=err, all_iteration_dis=all_iteration_dis,
                           node_in_Community=node_in_Community)

# 谣言溯源模型比对，后端服务
@app.route('/ComparisonOfRumorTraceabilityModel', methods=["GET", "POST"])
def ComparisonOfRumorTraceabilityModel():
    node_in_Community = GE.node_in_Community  # 每个节点所在的分区
    err1 = "false"  # 返回错误信息
    err2 = "false"  # 返回错误信息
    if request.method == "GET":
        # 初始化权重矩阵
        err1 = "true"
        err2 = "true"
        active_records1 = json.dumps([])
        active_records2 = json.dumps([])
        edge_records1 = json.dumps([])
        edge_records2 = json.dumps([])
        return render_template('ComparisonOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                               ObserverNodeList1=[], active_records1=active_records1,
                               edge_records1=edge_records1,
                               shortestPath1=[], err1=err1,err2=err2, node_in_Community=node_in_Community,
                               ObserverNodeList2=[], active_records2=active_records2,
                               edge_records2=edge_records2,
                               shortestPath2=[],all_iteration_dis1=[],all_iteration_dis2=[]
                               )
    else:  # request：请求对象，获取请求方式数据
        percentage1 = request.form.get('percentage1')  # 观测点数量
        iteration1 = request.form.get('iteration1')  # 迭代次数
        method1 = request.form.get('method1')  # 选择观测点方法
        err1=inputJudge(percentage1, iteration1, method1, err1)

    percentage2 = request.form.get('percentage2')  # 观测点数量
    iteration2 = request.form.get('iteration2')  # 迭代次数
    method2 = request.form.get('method2')  # 选择观测点方法
    err2 = inputJudge(percentage2, iteration2, method2, err2)

    if err1 != "false" or err2!="false":

        return render_template('ComparisonOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                               ObserverNodeList1=[], active_records1=json.dumps([]), edge_records1=json.dumps([]),
                               shortestPath1=[], err1=err1,err2=err2, node_in_Community=node_in_Community,
                               ObserverNodeList2=[], active_records2=json.dumps([]), edge_records2=json.dumps([]),
                               shortestPath2=[],all_iteration_dis1=[],all_iteration_dis2=[])
    ObserverNodeList1, active_records1, edge_records1, shortestPath1, all_iteration_dis1=SD(iteration1,float(percentage1),int(method1),1)
    ObserverNodeList2, active_records2, edge_records2, shortestPath2, all_iteration_dis2 = SD(iteration2,float(percentage2),int(method2), 2)
    return render_template('ComparisonOfRumorTraceabilityModel.html', graph_data=sd.graph_data,
                           ObserverNodeList1=ObserverNodeList1,
                           active_records1=active_records1, edge_records1=edge_records1,
                           shortestPath1=shortestPath1, err1=err1,err2=err2, all_iteration_dis1=all_iteration_dis1,
                           node_in_Community=node_in_Community,ObserverNodeList2=ObserverNodeList2,
                           active_records2=active_records2, edge_records2=edge_records2,
                           shortestPath2=shortestPath2, all_iteration_dis2=all_iteration_dis2)

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
