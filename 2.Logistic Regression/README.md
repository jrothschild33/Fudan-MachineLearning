# Homework 2: Logistic Regression——Classification

# Task

Please implement two models (probabilistic generative model && logistic regression model) to predict whether a person can make over 50k a year according to the personal information.

# Dataset

train.csv and test.csv have 32561 and 16281 samples respectively.

The attributions are: age, workclass, fnlwgt, education, education num, marital-status, occupation relationship, race, sex, capital-gain, capital-loss, hours-per-week, native-country, make over 50K a year or not.

# Attribute descriptions

- The symbol "?” denotes “unsure”
- age: continuous.
- workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
- fnlwgt: continuous. The number of people the census takers believe that observation represents.
- education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
- education-num: continuous.
- marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
- occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
- relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
- race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
- sex: Female, Male.
- capital-gain: continuous.
- capital-loss: continuous.
- hours-per-week: continuous.
- native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# Submitted files

- Source code (python): It should include at least two programming files, namely generative.py and logistic.py. Please implement the two models by yourself.
- submission.csv: You are required to submit your results according to the format of sampleSubmission.csv. The first row is (id, label), where id represents the sample order in the test set. (label =0 means “<= 50K” 、 label = 1 means “ >50K ”)
- Report.pdf: Please describe the configurations for running your code, and answer the following questions in your report with no more than 2 pages:

1. Please compare the accuracy of your implemented generative model and logistic regression. Which one is better?
2. Please implement input feature normalization and discuss its impact on the accuracy of your model.
3. Please implement regularization of logistic regression and discuss its impact on the accuracy of your model. (For regularization, please refer to “Lecture 1.1-Regression”)
4. Please discuss which attribute you think has the most impact on the results.

# Implementation

## 整体概述

1. 目标：分别用概率生成模型、逻辑回归模型预测年收入是大于 50K 还是小于 50K。
2. 数据描述：共计 32561 条数据，6 个定量特征，8 个定性特征

| 特征           | 类型                 | 类别/取值范围                            |
| -------------- | -------------------- | ---------------------------------------- |
| income         | 因变量（0-1 型标签） | <=50K（0）：7841 个，>50K（1）：24720 个 |
| age            | 定量特征             | 17~90 岁                                 |
| workclass      | 定性特征             | 8 个水平                                 |
| fnlwgt         | 定量特征             | 12285~ 1484705                           |
| education      | 定性特征             | 16 个水平                                |
| education-num  | 定量特征             | 1~16                                     |
| marital-status | 定性特征             | 7 个水平                                 |
| occupation     | 定性特征             | 14 个水平                                |
| relationship   | 定性特征             | 6 个水平                                 |
| race           | 定性特征             | 5 个水平                                 |
| sex            | 定性特征             | 2 个水平                                 |
| capital-gain   | 定量特征             | 0~ 99999                                 |
| capital-loss   | 定量特征             | 0~4356                                   |
| hours-per-week | 定量特征             | 1~99                                     |
| native-country | 定性特征             | 41 个水平                                |

## 数据处理

1. 填充缺失值：发现“workclass”一列中存在“ ？”数据，以众数数据“Private”进行填充。
2. 定量数据处理：归一化（Normalized），由于数据在实际中均大于 0，所以用归一化效果要好于标准化（Standardized），处理后的定量数据均处于[0,1]之间。
3. 定性数据处理：
   1. 因变量 income：标签化，<=50K 的数据归为“0”、>50K（1）的数据归为“1”
   2. 其他定性数据：处理为哑变量，8 个定性特性化为 101 个特征。

## 训练模型

### 1. 概率生成模型

1.计算方法：
$$\mu^{*}=\frac{1}{N} \sum_{n=1}^{N} x^{n}, \quad \Sigma^{*}=\frac{1}{N} \sum_{n=1}^{N}\left(x^{n}-\mu^{*}\right)\left(x^{n}-\mu^{*}\right)^{T}, \mu^{1}=\frac{1}{7841} \sum_{n=1}^{7841} x^{n}, \mu^{2}=\frac{1}{24720} \sum_{n=1}^{24720} x^{n}$$
$$\Sigma^{1}=\frac{1}{7841} \sum_{n=1}^{7841}\left(x^{n}-\mu^{*}\right)\left(x^{n}-\mu^{*}\right)^{T}, \quad \Sigma^{1}=\frac{1}{24720} \sum_{n=1}^{24720}\left(x^{n}-\mu^{*}\right)\left(x^{n}-\mu^{*}\right)^{T}, \quad \Sigma=P\left(C_{1}\right) \Sigma^{1}+P\left(C_{2}\right) \Sigma^{2}$$

$$
P\left(C_{1} \mid x\right)=\sigma(z)=\sigma(w x+b),\left\{\begin{array}{c}
w=\left(\mu^{1}-\mu^{2}\right)^{T} \Sigma^{-1} \\
b=-\frac{1}{2}\left[\left(\mu^{1}\right)^{T} \sum^{-1} \mu^{1}-\left(\mu^{2}\right)^{T} \Sigma^{-1} \mu^{2}\right]+\ln \frac{N_{1}}{N_{2}}
\end{array}\right.
$$

2.模型准确度：0.2408095574460244 3.总结：概率生成模型的分类效果较差

### 2. 逻辑回归模型

1.计算方法：

$$f_{w, b}(x)=\sigma\left(\sum_{i} w_{i} x_{i}+b\right), L(f)=\sum_{n} l\left(f\left(x^{n}\right), \hat{y}^{n}\right)$$
$$\text { Cross entropy: } l\left(f\left(x^{n}\right), \hat{y}^{n}\right)=-\left[\hat{y}^{n} \ln f\left(x^{n}\right)+\left(1-\hat{y}^{n}\right) \ln \left(1-f\left(x^{n}\right)\right)\right]$$
$$\text { L1 Penalty: } J(\theta)_{L 1}=J(\theta)+\lambda \sum_{j=1}^{n}\left|\theta_{j}\right|(j \geq 1), \text { L2 Penalty: } J(\theta)_{L 2}=J(\theta)+\lambda \mid \sum_{j=1}^{n}\left(\theta_{j}\right)^{2}(j \geq 1)$$

2.模型准确度：0.8523079757992691 3.总结：逻辑回归模型进行的分类较好

## 问题回答

1.Please compare the accuracy of your implemented generative model and logistic regression. Which one is better？

> 经过试验，逻辑回归的模型效果更好。

2.Please implement input feature normalization and discuss its impact on the accuracy of your model.

> 如图 1 所示，如果不经过归一化处理，模型准确度下降为 0.7957679432449863，且 L2 正则化方法下的准确度会下降更加严重。

3.Please implement regularization of logistic regression and discuss its impact on the accuracy of your model. (For regularization, please refer to “Lecture 1.1-Regression”)

> 由于模型中特征较多，转换为哑变量后总计有 107 个特征，故 L1 正则化方法会更好，经过试验（如图 2 所示），可以看出 L1 方法的准确性比 L2 方法要高，最终选取 L1 方法，$λ=1/0.8$，拟合出最终模型。经过 L1 正则化后模型参数减少至 81 个，模型的准确度为 0.8531371886612819，比未经过正则化的模型准确度 0.8523079757992691 提升了一些。
> ![learningcurve](https://zhoujianan.com/assets/school/FDU_Course_ML/2.Logistic%20Regression/learningcurve.png)

4.Please discuss which attribute you think has the most impact on the results.

> 对结果影响最大的参数为: capital_gain，其系数最大（30.100935272874228）
