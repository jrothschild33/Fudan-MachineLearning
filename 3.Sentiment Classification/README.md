# Homework 3: Sentiment Classification

# Task

This task is based on subtask 2 of SemEval-2014 Task 4: [Aspect Based Sentiment Analysis](http://alt.qcri.org/semeval2014/task4/)

You are required to implement two neural networks (RNN and CNN or their variants) for sentiment classification specific to an aspect.

For example:

- _“Even though its good seafood, the prices are too high”_.
- This sentence contains two aspects, namely “seafood” and “prices”. The sentiment for the two aspects are positive and negative respectively.

# Provided files

`laptop-train.txt`, `laptop-test.txt`

Each line contains id, text, aspect:start-end, label separated by tabs:

- id: the id
- text: the text content
- aspect:start-end: “aspect” denotes the aspect term mentioned in the text, and “start-end” denotes the start and end character index of the aspect term occurred in the text
- label: the sentiment label (positive, negative and neutral) specific to an aspect

`eval.py`

This file can help you evaluate your model’s accuracy and Macro-F1 score.

# Submitted files

- **Source code (pytorch)**: It should include at least two programming files, namely RNN.py and CNN.py. You can use the variants of RNN and CNN models.
- **result_rnn.txt, result_cnn.txt** : Your job is to predict the aspect-based sentiment polarity for each instance in the test files, and write your predicted labels in result files. The labels in test files are only for evaluation.

  > Note: The result files only need to contain the label for each instance in the test set (a label each line). Please follow the instance order in the test files for ease of evaluation.

- **Report.pdf**: Please describe the configurations for running your code, and provide some analysis for the results in your report with no more than 2 pages.

# Implementation

## 整体概述

1.目标：分别使用 CNN、RNN 相关模型，对 Laptop 语料库进行分析，根据每个句子中的不同方面（Aspect）进行情感分析，分别有 3 类标签“Positive（积极）、Negative（消极）、Neutral（中性）”，一个句子中可能包含多个方面，需要进行处理。

2.数据描述：

1. 训练数据：共有 2358 条文本数据，删去 conflict 后剩余 2313 条数据
2. 测试数据：共有 654 条文本数据，删去 conflict 后剩余 638 条数据

## 试验思路

1.窗口上下文：如句子[W1,W2,W3,W4,W5,W6,W7,...]，W3 为情感对象，我们这里可以去窗口大小为 2 的上下文[W1,W2,W3, W4, W5]作为相关文本。

2.文法关联词：窗口上下文的问题在于对情感对象表达了观点的词极可能在位置上离情感对象很远，如果使用文法分析，对 W3 的上下文 W1~W5，各种提取了其在文法树上有关系的单词。如下图：
![wordprocess](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/wordprocess.png)
通过文法分析，screen 这个单词和 the 与 good 具备依存关系，那么我们把它们提取出来做相关文本。但对 screen 做依存词抽取肯定是不够的，这里我对 I, think, the, scree ,is ,good ,but 这 7 个单词都抽取了依存词作为相关文本。

## Text-CNN 模型

1.整体架构：
![Text-CNN](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/Text-CNN.png)

2.模型效果：
![result1](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/result1.png)

3.整体效果：acc: 0.42946708463949845、f1: 0.31973015040107694

4.模型评价：模型整体准确率不够理想，可能由于训练数据较少，可以适当增加训练数据的数量，辅助外部大型语料库数据进行训练，也可能是由于对 Aspect 的提取技术仍需要提升。

## Text-CNN 模型

1.整体架构：
![RNN](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/RNN.png)

2.模型效果：
![result2](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/result2.png)

3.整体效果：acc: 0.5344827586206896, f1: 0.23220973782771537

4.模型评价：RNN 的模型准确率要高于 Text-CNN 模型，但是期 F1-Score 分数低于 Text-CNN 模型，从结果上看把所有类别均分到了 Positive 类下，可能存在某些分类上的问题。两个模型通用的问题即为训练数据量较少，从常见的 NLP 模型如 BERT 来看，训练数据集大小至少要 5GB 以上，准确度才会有显著提升。
