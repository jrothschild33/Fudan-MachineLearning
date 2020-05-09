# -*- coding: gbk -*-
# author: 周嘉楠

# 导入所用的库
import sys
import csv
import numpy as np
import pandas as pd
import seaborn as sns
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
from datetime import datetime
from sklearn import metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def readMyFile():   # 读取数据并填充空缺值
    train = pd.read_csv('train.csv', parse_dates=['Date'])
    test = pd.read_csv('test.csv')
    train.replace(['NR'], [0.0], inplace=True)
    test.replace(['NR'], [0.0], inplace=True)
    return train, test

def myStandardize(file):                                                    # 数据值处理:归一化
    cols = list(file.columns)                                               # 提取原文件中列名
    vals = []                                                               # 准备空列表，填充归一化后的数据
    for i in cols[2:]:                                                      # 从原文件第2列后开始处理
        x = file.loc[:, i]                                                  # 选中第i列
        new_x = list(preprocessing.scale(x))                                # 逐列归一化（Standardize）
        vals.append(new_x)                                                  # 将归一化的列填充到vals列表中
    vals = np.array(vals).transpose()                                       # 将归一化的列表转换成array并转置
    vals = pd.DataFrame(vals)                                               # 将上一步得到的array变成dataframe
    cols_id = file.iloc[:, :2]                                              # 提取原文件中前2列编号名称
    new_file = pd.merge(cols_id, vals, left_index=True, right_index=True)   # 将归一化数据与编号合并成新表
    return new_file                                                         # 查看归一化后的新数据表


def extractFeaLab(file_1, file_2, file_3):                                  # 提取训练feature、训练label、测试feature
    my_indexs1 = file_1.iloc[:, 0].drop_duplicates()
    my_indexs2 = file_3.iloc[:, 0].drop_duplicates()
    train_X = []
    train_Y = []
    test_X = []
    for id in my_indexs1:
        fea_array1 = np.array(file_1[file_1.iloc[:, 0] == id].iloc[:, 2:], dtype=np.float32)
        lab_array1 = np.array(file_2[file_2.iloc[:, 0] == id].iloc[:, 2:], dtype=np.float32)
        for hour in range(24 - 8):
            fea = fea_array1[:, hour:hour + 8].flatten()                    # 将array平坦化（降维）
            label = lab_array1[9, hour + 8].flatten()                       # 将array平坦化（降维）
            train_X.append(fea)
            train_Y.append(label)

    for id in my_indexs2:
        fea_array2 = np.array(file_3[file_3.iloc[:, 0] == id].iloc[:, 2:], dtype=np.float32)
        fea2 = fea_array2.flatten()                                         # 将array平坦化（降维）
        test_X.append(fea2)

    train_X_ = np.array(train_X)
    train_Y_ = np.array(train_Y)
    test_X_ = np.array(test_X)
    # print(train_X_.shape)                       # (16*240=3840,18*8=144)：由new_train得到3840个144维特征
    # print(train_Y_.shape)                       # (3840,1)：由train得到3840个PM2.5原预测值
    # print(test_X_.shape)                        # (240,18*8=144)：有240个待输入的特征，来预测test中的PM2.5值
    return train_X_, train_Y_, test_X_

def myLinearReg(train_X_, train_Y_,test_X_):    # 在训练集上训练模型，并划分16:1验证集进行验证，计算RMSE
    X_train, X_test, y_train, y_test = train_test_split(train_X_, train_Y_, test_size=240 / 3840, random_state=123)

    print('训练集测试及参数:')
    print(' X_train.shape={}\n y_train.shape ={}\n X_test.shape={}\n y_test.shape={}'
          .format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    linreg = LinearRegression()
    model = linreg.fit(X_train, y_train)
    print('模型参数:\n', model)
    print('模型截距:\n', linreg.intercept_)
    print('参数权重:\n', linreg.coef_)

    y_pred = linreg.predict(X_test)
    sum_mean = 0
    for i in range(len(y_pred)):
        sum_mean += (y_pred[i] - y_test[i]) ** 2
    sum_erro = np.sqrt(sum_mean / len(y_pred))
    print("RMSE(在验证集中):", sum_erro)

    # 画出验证集预测情况
    plt.figure()
    plt.plot(range(len(y_pred)), y_pred, 'b', linestyle='--', label="predict")
    plt.plot(range(len(y_pred)), y_test, 'r', linestyle='solid', label="test")
    plt.legend(loc = "best")
    plt.xlabel("Number of test data")
    plt.ylabel('PM2.5')
    plt.show()

    # 将训练好的模型用在test数据上进行预测
    y_pred2 = linreg.predict(test_X_)

    # 画出预测情况
    plt.figure()
    plt.plot(range(len(y_pred2)), y_pred2, 'g', linestyle='--', label="Real predict")
    plt.legend(loc="best")
    plt.xlabel("id")
    plt.ylabel('Real predict PM2.5')
    plt.show()

    # 将结果存储数据到sampleSubmission.csv
    result = pd.DataFrame(y_pred2)
    result.columns = ['values']
    id_names = []
    for i in range(len(result)):
        id_names.append('id_%d' % i)
    result.index = id_names
    result.index.name = 'id'
    result.to_csv('sampleSubmission.csv')