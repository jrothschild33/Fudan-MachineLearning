# -*- coding: gbk -*-
# author: 周嘉楠

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import linear_model

'1. 数据准备 ============================================================================'

# 1. 数据准备
# 1.1 导入数据
def import_my_train_file(file):
    # 导入数据
    lines = np.loadtxt(file, delimiter=',', dtype='str')
    print('数据集大小:', lines.shape[0] - 1)
    print('数据集特征:', lines.shape[1] - 1)
    print('数据集特征列表:', lines[0])

    # 转换为DataFrame
    data = pd.DataFrame(lines[1:], columns=lines[0])

    # 查看定量变量
    quant_var = list(data.iloc[:, [0, 2, 4, 10, 11, 12]].columns)
    print('定量特征有：', len(quant_var), '个')
    print('定量特征为：', quant_var)
    print('定量特征描述：\n', data.iloc[:, [0, 2, 4, 10, 11, 12]].astype('float').describe())

    # 查看定性变量
    quali_var = list(data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13]].columns)
    print('定性特征有：', len(quali_var), '个')
    print('定性特征为：', quali_var)
    print('定性特征描述：\n', data.iloc[:, [1, 3, 5, 6, 7, 8, 9, 13]].describe())

    # 对表格列顺序进行调整
    my_cols = list(data.columns)[:-1]
    my_cols.insert(0, 'income')
    data = data[my_cols]

    return data, lines, quant_var, quali_var


# 1.2 数据处理：缺失值处理；将定量变量归一化，将定性变量转换为哑变量
def clean_my_data(data, quant_var, quali_var):
    # 查看每列数据是否有缺失值：没有缺失值，将" ？"进行处理
    # data.info()

    # 使用众数填充" ？"
    imp_mode = SimpleImputer(missing_values=' ?', strategy='most_frequent')
    data['workclass'] = imp_mode.fit_transform(np.array(data['workclass']).reshape(-1, 1))

    # 对定量变量做归一化处理
    data[quant_var] = MinMaxScaler().fit_transform(data[quant_var])

    # 定性变量(多个水平）做哑变量处理，8个定性变量经过处理后得到101个特征
    dummy = pd.get_dummies(data.loc[:, quali_var])

    # 将income一列化为0-1型变量
    data['income'] = data['income'].map({' <=50K': 0, ' >50K': 1})

    # 将处理好的定性变量与原表格拼接，并删去原表格无用列
    newdata = pd.concat([data, dummy], axis=1)
    newdata.drop(quali_var, axis=1, inplace=True)

    return newdata, dummy


# 1.3 提取处理好的数据
def extract_my_data(newdata):
    # 提取表格中的特征features、标签labels
    x_total = np.array(newdata.iloc[:, 1:].astype('float'))
    y_total = np.array(newdata.iloc[:, :1].astype('float')).flatten()  # 将y_total变为行向量

    # 标签分类定位
    pos_index = np.where(y_total == 1)  # 定位出所有y_total为1的索引值（pos_index属性为tuple）
    neg_index = np.where(y_total == 0)  # 定位出所有y_total为0的索引值（neg_index属性为tuple）

    return x_total, y_total, pos_index, neg_index

'2. 逻辑回归 ============================================================================'

# 2. 逻辑回归模型
# 2.1 无正则化情况下的模型
def my_logi_regression(x_total, y_total):
    # 训练模型
    # 实例化lr_clf，用来表示逻辑回归模型
    lr_clf = linear_model.LogisticRegression()
    # 调用fit函数，输入训练数据x_total, y_total，对模型进行训练
    lr_clf.fit(x_total, y_total)
    # coef_[0]：模型参数w_i
    print(lr_clf.coef_[0])
    # intercept_：模型截距项b
    print(lr_clf.intercept_)

    # 使用模型预测
    # 调用predict函数：将x_total输入训练好的模型，得到预测标签y_pred
    y_pred = lr_clf.predict(x_total)
    # 若y_pred == y_total则为1，否则为0，计算其均值，作为模型准确率
    print('准确率(未正则化):', (y_pred == y_total).mean())


# 2.2 有正则化情况下的模型: 分别做L1、L2正则化，并画图选择最优参数
def find_best_logi_arg(x_total, y_total):
    l1 = []
    l2 = []
    l1test = []
    l2test = []
    # 划分训练集、测试集（7：3）
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x_total, y_total, test_size=0.3, random_state=420)

    for i in np.linspace(0.05, 1, 19):
        lrl1 = LR(penalty="l1", solver="liblinear", C=i, max_iter=1000)
        lrl2 = LR(penalty="l2", solver="liblinear", C=i, max_iter=1000)

        lrl1 = lrl1.fit(Xtrain, Ytrain)
        l1.append(accuracy_score(lrl1.predict(Xtrain), Ytrain))
        l1test.append(accuracy_score(lrl1.predict(Xtest), Ytest))

        lrl2 = lrl2.fit(Xtrain, Ytrain)
        l2.append(accuracy_score(lrl2.predict(Xtrain), Ytrain))
        l2test.append(accuracy_score(lrl2.predict(Xtest), Ytest))

    graph = [l1, l2, l1test, l2test]
    color = ["green", "black", "lightgreen", "gray"]
    label = ["L1", "L2", "L1test", "L2test"]

    # 画出学习图
    # 结果说明：选择L1正则化，正则化强度倒数C选择0.8，这样的模型效果最好
    plt.figure(figsize=(6, 6))
    for i in range(len(graph)):
        plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
    plt.legend(loc=4)
    plt.show()

# 2.3 拟合最优模型
def my_best_logi_model(newdata, x_total, y_total):
    # L1正则化训练模型
    lrl1 = LR(penalty="l1", solver="liblinear", C=0.8, max_iter=1000)
    lrl1 = lrl1.fit(x_total, y_total)
    print('模型准确度：', accuracy_score(lrl1.predict(x_total), y_total))
    print('模型系数：\n', lrl1.coef_)
    print('降维后参数有：', (lrl1.coef_ != 0).sum(axis=1), '个')

    # 找出系数最大的参数（即为对结果影响最大的参数）
    print('对结果影响最大的参数为:', newdata.columns[lrl1.coef_.argmax() + 1])
    print('对结果影响最大的参数的系数为:', lrl1.coef_.flatten()[lrl1.coef_.argmax()])
    return lrl1


'3. 预测结果 ============================================================================'

# 3.1 导入测试集数据，并进行处理

def process_my_test_file(file, quant_var, quali_var, dummy):
    # 导入test数据集，进行数据处理
    lines = np.loadtxt(file, delimiter=',', dtype='str')
    print('测试集数据大小:', lines.shape[0] - 1)

    # 转换为DataFrame
    data = pd.DataFrame(lines[1:], columns=lines[0])

    # 使用众数填充" ？"
    imp_mode = SimpleImputer(missing_values=' ?', strategy='most_frequent')
    data['workclass'] = imp_mode.fit_transform(np.array(data['workclass']).reshape(-1, 1))

    # 对定量变量做归一化处理
    data[quant_var] = MinMaxScaler().fit_transform(data[quant_var])

    # 定性变量(多个水平）做哑变量处理，8个定性变量经过处理后得到100个特征（少了一个），需要进一步处理
    dummy2 = pd.get_dummies(data.loc[:, quali_var])

    # 查找新数据中是否缺少特征：
    if len(dummy2.columns) != len(dummy.columns):
        testls = []
        for i in list(dummy.columns):
            if i in list(dummy2.columns):
                pass
            else:
                testls.append(i)
        # 如果缺失，则将缺失列在原数据中索引出来，并插入新数据中与原数据相同的列位置
        for i in range(len(testls)):
            dummy2.insert(
                # 获取缺失元素在原列表中的索引值
                list(dummy.columns).index(testls[i]),
                # 添加列名，将原数据的指定列插入缺失列
                testls[i], dummy[testls[i]]
            )

    # 将处理好的定性变量与原表格拼接，并删去原表格无用列
    newdata = pd.concat([data, dummy2], axis=1)
    newdata.drop(quali_var, axis=1, inplace=True)

    # 将处理好的数据转换为array形式
    x_test = np.array(newdata.astype('float'))

    return x_test

# 3.2 将处理好的数据输入训练好的模型，得到预测结果

def pred_my_data(lrl1, x_test):
    y_test_pred = lrl1.predict(x_test)
    # 将y_test_pred结果保存成csv
    y_test_pred = pd.DataFrame(y_test_pred)
    y_test_pred.columns = ['label']
    y_test_pred.index.name = 'id'
    y_test_pred.to_csv('submission_LR.csv')

if __name__ == '__main__':
    print('1.数据准备:')
    print('-----------------------------------------------------------------------------')
    print('1.1 导入数据:')
    print('-----------------------------------------------------------------------------')
    data, lines, quant_var, quali_var = import_my_train_file('train.csv')
    print('1.2 数据处理：处理缺失值；定量变量归一化，定性变量转换为哑变量:')
    print('-----------------------------------------------------------------------------')
    newdata, dummy = clean_my_data(data, quant_var, quali_var)
    print('1.3 提取处理好的数据:')
    print('-----------------------------------------------------------------------------')
    x_total, y_total, pos_index, neg_index = extract_my_data(newdata)
    print('2. 逻辑回归模型:')
    print('2.1 无正则化情况下的模型:')
    print('-----------------------------------------------------------------------------')
    my_logi_regression(x_total, y_total)
    print('2.2 有正则化情况下的模型:分别做L1、L2正则化，并画图选择最优参数')
    print('-----------------------------------------------------------------------------')
    find_best_logi_arg(x_total, y_total)
    print('2.3 拟合最优模型:')
    print('-----------------------------------------------------------------------------')
    lrl1 = my_best_logi_model(newdata, x_total, y_total)
    print('3.1 导入测试集数据，并进行处理:')
    print('-----------------------------------------------------------------------------')
    x_test = process_my_test_file('test.csv', quant_var, quali_var, dummy)
    print('3.2 将处理好的数据输入训练好的模型，得到预测结果:"submission_LR.csv"')
    print('-----------------------------------------------------------------------------')
    pred_my_data(lrl1, x_test)