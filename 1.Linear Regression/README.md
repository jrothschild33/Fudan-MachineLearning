# Homework 1: PM 2.5 prediction
# Task
The datasets are real observations downloaded from the website of the Central Meteorological Administration. Please use linear regression to predict the PM2.5 value.

# Dataset
* train.csv：The observations of the first 20 days of each month are used for training.
* test.csv：The observations of 9 consecutive hours are taken from the remaining 10 days of each month. All observations for the first 8 hours are considered as features, and PM2.5 at the 9th hour is used as the answer. A total of 240 unique samples were taken out for testing. Please predict the PM2.5 of these 240 samples according to features.

# Submitted files
* Source code (python)
* submission.csv: You are required to submit your results according to the format of sampleSubmission.csv.
* Report.pdf: Please briefly describe your method and configurations for running your code with 1-2 pages.
* Evaluation: We will evaluate your results with the RMSE (root mean square error) metric. You can sample some data from the training.csv for testing offline.

# Implementation

## 数据处理
1.导入数据`train.csv`和`test.csv`，发现RAINFALL存在空缺值，以数值0填充空缺值。
```python
def readMyFile():   # 读取数据并填充空缺值
    train = pd.read_csv('train.csv', parse_dates=['Date'])
    test = pd.read_csv('test.csv')
    train.replace(['NR'], [0.0], inplace=True)
    test.replace(['NR'], [0.0], inplace=True)
    return train, test
```

2. 观察`test.csv`中的每日数据维度为18*8，即用144个特征预测第9个小时的PM2.5值，所以需要将`train.csv`中每日数据维度进行处理：
![dataprocessing1](https://jrothschild.oss-cn-shanghai.aliyuncs.com/FDU_Course_ML/1.Linear%20Regression/dataprocessing1.png)
![dataprocessing2](https://jrothschild.oss-cn-shanghai.aliyuncs.com/FDU_Course_ML/1.Linear%20Regression/dataprocessing2.png)

3.由于各个特征值之间数值相差较大，无法直接回归，需要对数据进行归一化处理（Standardize）

$$\widehat{x}_{i}=\frac{x_{i}-\mu}{\sigma}$$

```python
def myStandardize(file):         # 数据值处理:归一化
    cols = list(file.columns)    # 提取原文件中列名
    vals = []                       # 准备空列表，填充归一化后的数据
    for i in cols[2:]:            # 从原文件第2列后开始处理
        x = file.loc[:, i]        # 选中第i列
        new_x = list(preprocessing.scale(x))  # 逐列归一化（Standardize）
        vals.append(new_x)         # 将归一化的列填充到vals列表中
    vals = np.array(vals).transpose()  # 将归一化的列表转换成array并转置
    vals = pd.DataFrame(vals)    # 将上一步得到的array变成dataframe
    cols_id = file.iloc[:, :2]   # 提取原文件中前2列编号名称
    new_file = pd.merge(cols_id, vals, left_index=True, right_index=True)  
    # 将归一化数据与编号合并成新表
    return new_file                 # 查看归一化后的新数据表
```

4.提取表格中的数据，并进行降维处理，以方便输入线性模型进行训练：
   1. 训练x值（train_X）：从归一化后的new_train文件中提取，用于训练模型
   2. 训练y值（train_Y）：从未处理的train文件中提取，用于训练模型（因为需要预测得到PM2.5的原数值，而不是归一化后的值）
   3. 测试x值（test_X）：从归一化后的new_test中提取，使用训练好的模型进行预测
```python
def extractFeaLab(file_1, file_2, file_3):      # 提取训练feature、训练label、测试feature
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
```

## 模型构建
1.将处理好的数据输入线性模型：
$$y_{i}=w_{0}+\sum_{i=1}^{144} w_{i} x_{i}$$

```python
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
```

* 输出结果：

| 类别           | 代码                                                                                           |
| ---------------- | ------------------------------------------------------------------------------------------------ |
| 训练集测试及参数 | X_train.shape=(3600, 144),y_train.shape =(3600, 1),X_test.shape=(240, 144),y_test.shape=(240, 1) |
| 模型参数     | LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)                  |
| 模型截距     | [134.25134]                                                                                      |
| RMSE(在验证集中) | [7.1660585]                                                                                      |

2.将模型的训练预测效果可视化：
```python
# 画出验证集预测情况
plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', linestyle='--', label="predict")
plt.plot(range(len(y_pred)), y_test, 'r', linestyle='solid', label="test")
plt.legend(loc = "best")
plt.xlabel("Number of test data")
plt.ylabel('PM2.5')
plt.show()
```

![Figure_1](https://jrothschild.oss-cn-shanghai.aliyuncs.com/FDU_Course_ML/1.Linear%20Regression/Figure_1.png)

3.将训练好的模型用于new_test数据集中提取的特征x_i，进行预测PM2.5，并将数据保存至sampleSubmission.csv
```python
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
```

![Figure_2](https://jrothschild.oss-cn-shanghai.aliyuncs.com/FDU_Course_ML/1.Linear%20Regression/Figure_2.png)
