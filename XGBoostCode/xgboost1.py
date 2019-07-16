#!usr/bin/env python  
#-*- coding:utf-8 _*-  
""" 
@author:JiaYiHe 
@file: xgboost1.py 
@time: 2019/07/{DAY} 
"""

'''
此文件可以用于训练模型与预测，可输出mse，mae，accuracy，对binary问题还可输出precision，recall

1.读取configfile，读入输入输出文件路径；
2.训练模型；
3.预测；

每次训练模型会输出对应评估参数mse，mae，accuracy等
每次训练模型会输出重要影响因子排序
如果输入数据集存在分类，则需要输入分类与特征集dict.csv

要求输入数据第一列为key，第二列为分类结果（可多分类，但需要是数值），后续列为特征
分类结果与特征值为数值，要求在进入程序前把特征补全，不能含空值
'''

from sklearn.cross_decomposition import PLSRegression  #PLS算法
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import warnings
from sklearn.externals import joblib
import datetime
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
import math
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score      # 准确率
from sklearn.metrics import precision_score     # 精准率
from sklearn.metrics import recall_score        # 召回率
from sklearn.tree import DecisionTreeRegressor
#from sklearn.metrics import zero_one_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
import time

#配置文件路径
configPath=''

#数据路径与输出结果路径
dataPath=''
resultPath=''
predictDataPath=''
header_flag=False

label1=''
label2=''
label3=''
n_Components=0
maxIter=0


#数据
trainX=[]
trainY=[]
testX=[]
testY=[]
header_nm=[]

#评价指标
# vars = {}.fromkeys(['mae', 'mse'])
vars={}

#xgboost参数
params = {
            'booster': 'gbtree',
            'objective': 'reg:gamma',
            'gamma': 0.1,
            'max_depth': 6,
            'lambda': 3,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'min_child_weight': 3,
            'silent': 1,
            'eta': 0.1,
            'seed': 1000,
            'nthread': 4,
        }

#time
currentTime = time.strftime("%Y-%m-%d_%H-%M", time.localtime())

def initial():
    global configPath
    print("Please input your configfile Path:\XGBoostCode下寻找configfile,Y/N: ")  # 读入配置文件路径
    path = input()
    if path == 'y' or path == 'Y':
        folderPath = r'C:\Users\JiaYiHe\Documents\...\XGBoostCode'
    else:
        folderPath = input('reinput configfile path:')

    print('configfile Path is： ', folderPath.strip())  # 显示出配置文件路径
    configPath = folderPath.strip()

# 写入Log 信息函数
def writeLog(path,content):
    f=open(path,'a',encoding='utf-8')
    f.write(content+'\n')
    f.close()

# 读取配置信息
def readConfigFile():
    global configPath
    global dataPath
    global n_Components
    global resultPath
    global maxIter
    global label1
    global label2
    global label3
    global currentTime
    global header_flag

    f = open(configPath+'/'+'configFile.txt', 'r',encoding='utf-8')
    line = f.readline().replace('\n', '')
    while line:
        if '=' in line:
            temp = line.split('=')
            if temp[0]=='﻿dataPath':            #训练集路径
                dataPath=temp[1]
            elif temp[0]=='resultPath':         #输出结果，模型路径
                resultPath=temp[1].replace("\\","/")+'/'+currentTime
            elif temp[0]=='predictDataPath':    #测试集路径
                predictDataPath=temp[1]
            elif temp[0]=='header':
                if temp[1]=='True' or temp[1]=='TRUE' or temp[1]=='true': header_flag = True
                else: header_flag = False
            elif temp[0]=='n_components':
                n_Components=int(temp[1])
            elif temp[0]=='maxiter':
                maxIter=int(temp[1])
            elif temp[0]=='label':
                subStr=temp[1].split('$')
                label1=int(subStr[0])
                label2=subStr[1]
                label3=float(subStr[2])
        line = f.readline().replace('\n', '')  # 循环读取
    f.close()

    # 配置data路径
    print('datapath:'+dataPath)
    a=input('u can choose an other datapath:\n 1.C:\\Users\\...\\XGBoostCode\\data\\data_all.csv'
            '\n 2.C:\\Users\\...\\XGBoostCode\\data\\2.csv\n'
            '3. input new data path')
    if a==1 or a=='1':
        dataPath=r'C:\Users\...\XGBoostCode\data\data_all.csv'
    elif a==2 or a=='2':
        dataPath=r'C:\Users\...\XGBoostCode\data\2.csv'
    else:
        dataPath=input('dataPath: ')


#读取数据特征,cate用来对不同类型建立模型，默认所有数据均属于同一类型，无需处理；value用于记录某cate下的特征名称；
def readFeature(inputPath, cate=0, value=[]):
    global trainX, trainY, header_nm, header_flag

    trainX, trainY=[],[]
    ID, header_nm, flag=[],[], 0

    f = open(inputPath, 'r', encoding='utf-8')
    line = f.readline().replace('\n', '')

    # 如果输入的数据可区分类型，则只取此类型的X值
    if cate==0:
        while line:
            newline = line.replace("\t", ",")
            if ',' in newline:
                temp = line.split(',')

                # 忽略前2列，从第3列开始取作特征值
                # 如果输入数据存在header，读取特征值的名称
                if header_flag and flag == 0:
                    for i in range(2, len(temp)):
                        header_nm.append(temp[i])
                    flag += 1
                    line = f.readline().replace('\n', '')  # 循环读取
                    continue

                # 备用字段
                ID.append(temp[0])
                ftemp = []

                # 读取特征值
                for i in range(2,len(temp)):
                    ftemp.append(float(temp[i]))

                trainX.append(ftemp)  # 写入特征X
                y = float(temp[1])  # 写入标签y
                trainY.append(y)  # means R

                line = f.readline().replace('\n', '')  # 循环读取
            else:
                break

        trainX = pd.DataFrame(trainX)
        if len(header_nm)!=0:   #如果存在header，为训练集的特征名称赋值
            trainX.columns = header_nm

    # 如果输入的数据不区分类型，则取所有数据作为X
    else:
        while line:
            newline= line.replace("\t",",")
            if ',' in newline:
                temp = line.split(',')

                # 忽略前2列，从第3列开始取作特征值
                # 如果输入数据存在header，读取特征值的名称
                if header_flag and flag==0:
                    for i in range(2,len(temp)):
                        header_nm.append(temp[i])
                    flag+=1
                    line = f.readline().replace('\n', '')  # 循环读取
                    continue

                if temp[0] == cate:
                    ID.append(temp[0])
                    ftemp = []

                    # 读取特征值
                    for i in value:
                        ftemp.append(float(temp[header_nm.index(i)+2]))

                    trainX.append(ftemp)  #写入特征X
                    y = float(temp[1])  # 写入标签y
                    trainY.append(y)

                line = f.readline().replace('\n', '')      # 循环读取

            else:
                break
        f.close()
        trainX = pd.DataFrame(trainX)
        if len(value)!=0:   #如果存在header，为训练集的特征名称赋值
            trainX.columns = value


def estimate(X_train, Y_train, test_size=0.2, random_state=0):
    global params, vars

    ndarryX = np.array(X_train)
    ndarryy = np.array(Y_train)

    X_train, X_test, y_train, y_test = train_test_split(ndarryX, ndarryy, test_size=test_size, random_state=random_state)

    dtrain = xgb.DMatrix(X_train, y_train)
    num_rounds = 300
    plst = params.items()
    model = xgb.train(plst, dtrain, num_rounds)

    # 对测试集进行预测
    dtest = xgb.DMatrix(X_test)
    y_pred = model.predict(dtest)

    # 计算mse, accuracy_score
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    accuracy = accuracy_score(y_test, [int(round(x)) for x in y_pred])
    # precision = precision_score(y_test, [int(round(x)) for x in y_pred], average='samples')   #二分类问题
    # recall = recall_score(y_test, [int(round(x)) for x in y_pred])                            #二分类问题

    vars['mse'],vars['mae'],vars['accuracy'] = mse, mae, accuracy

    print(vars)
    if os.path.exists(resultPath) is False:
        os.makedirs(resultPath)
    writeLog(resultPath+'/'+'log.txt',str(vars))
    print('finished estimating!')

def createModel(X_train=trainX, Y_train=trainY, title=''):
    global resultPath, params, vars
    if len(trainX)==0 or len(trainY)==0:
        print("please input train data")
    else:
        # print(X_train)
        dtrain = xgb.DMatrix(X_train, Y_train)
        num_rounds = 300
        plst = params.items()
        model = xgb.train(plst, dtrain, num_rounds)
        print("finished modeling!")

        flag=input("是否保存模型y/n: ")
        if flag=='y' or flag=='Y':
            if os.path.exists(resultPath) is False:
                os.makedirs(resultPath)
            joblib.dump(model, resultPath + '/' + 'pls_'+title+'.m')  # 输出模型

        #输出评估结果
        estimate(X_train, Y_train)

        #显示特征值重要性
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plot_importance(model)
        plt.title(title+',评估参数：'+str(vars))
        # plt.text(0,-3.5,'评估参数：'+str(vars))
        plt.show()

def predict(cate=0, value=[]):
    global resultPath, predictDataPath, params, header_nm, trainX, trainY

    modelPath=input("model Path:")
    model = joblib.load(modelPath)
    predictDataPath=input("predict data path: ")
    readFeature(predictDataPath, header_flag, cate, value)

    dtest = xgb.DMatrix(trainX)
    y_pred = model.predict(dtest)

    xlsfile1 = resultPath+'/'+'predict_result'+'_' + cate + '.csv'
    df=trainX
    df['y_true']=trainY
    df['y_pred'] = pd.DataFrame(y_pred)

    df.to_csv(xlsfile1, index=False, encoding="utf_8_sig")
    print('finished prediction!')



if __name__=="__main__":
    warnings.filterwarnings("ignore")   #忽略警告
    initial()

    readConfigFile()         #读取配置文件信息

    # 当数据不存在分类时启用------------------------------------------------------------------------------------
    readFeature(dataPath)
    createModel(trainX, trainY)
    # predict()

    #------------------------------------------------------------------------------------------------------



    # 当数据存在多个分类时启用---------------------------------------------------------------------------------
    # xlfile = r'C:\Users\JiaYiHe\Documents\项目\伊利\伊利大数据项目三期\02需求分析\02客户提供资料\原奶预测模型\XGBoostCode\data\dict.csv'    # 读取不同类型的特征值
    # data = pd.read_csv(xlfile, delimiter=',', encoding='utf_8_sig', engine='python')
    # data = data.dropna()
    # data = {col: data[col].tolist() for col in data.columns}
    # for cate3 in data.keys():
    #     value=data[cate3]   #读每个特征值
    #     if os.path.exists(resultPath) is False:
    #         os.makedirs(resultPath)
    #     writeLog(resultPath + '/' + 'log.txt' , cate3)
    #     writeLog(resultPath + '/' + 'log.txt' , str(value))
    #     readFeature(dataPath, cate3, value)    #读取数据
    #     createModel(trainX, trainY, cate3)
    #     writeLog(resultPath + '/' + 'log.txt', '----------------------------------')
    #     # predict(cate3, value)

    #----------------------------------------------------------------------------------------------------


