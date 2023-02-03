# Fudan University-Machine Learning

2020 Spring Fudan University Machine Learning Course HW by prof. Chen Qin

# 1. Linear Regression

## Task

The datasets are real observations downloaded from the website of the Central Meteorological Administration. Please use linear regression to predict the PM2.5 value.

## Modules

- numpy==1.18.3
- pandas==0.25.3
- seaborn==0.10.1
- matplotlib==3.2.1
- sklearn.model_selection.train_test_split
- sklearn.metrics.mean_squared_error
- sklearn.linear_model.LinearRegression

## Results

![result](https://zhoujianan.com/assets/school/FDU_Course_ML/1.Linear%20Regression/result.png)

# 2. Logistic Regression

## Task

Implement two models (probabilistic generative model && logistic regression model) to predict whether a person can make over 50k a year according to the personal information.

## Modules

- numpy==1.18.3
- pandas==0.25.3
- seaborn==0.10.1
- matplotlib==3.2.1
- sklearn.preprocessing
- sklearn.preprocessing.MinMaxScaler
- sklearn.preprocessing.LabelEncoder
- sklearn.linear_model.LogisticRegression
- sklearn.metrics.accuracy_score
- sklearn.model_selection.train_test_split

## Results

![learningcurve](https://zhoujianan.com/assets/school/FDU_Course_ML/2.Logistic%20Regression/learningcurve.png)

# 3. Sentiment Classification

## Task

This task is based on subtask 2 of SemEval-2014 Task 4: [Aspect Based Sentiment Analysis](http://alt.qcri.org/semeval2014/task4/)

You are required to implement two neural networks (RNN and CNN or their variants) for sentiment classification specific to an aspect.

For example:

- _“Even though its good seafood, the prices are too high”_.
- This sentence contains two aspects, namely “seafood” and “prices”. The sentiment for the two aspects are positive and negative respectively.

## Modules

- numpy==1.18.3
- pandas==0.25.3
- torch==1.2.0
- torch.optim
- torch.nn.functional

## Results

![result1](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/result1.png)

![result2](https://zhoujianan.com/assets/school/FDU_Course_ML/3.Sentiment%20Classification/result2.png)

# 4. Auto Encoder

## Task

Please write an auto-encoder for the images.

- Use the trained encoder to obtain the 2-dimensional code of the **last 1000 images** in the test set, and visualize them with a scatterplot where different colors represent different digits.
- Use the decoder to generate 20 images by sampling some codes.

## Modules

- numpy == 1.18.3
- scipy == 1.2.1
- Pillow == 7.1.2
- tensorflow == 1.15.3
- torch == 1.2.0

## Results

![epoch](https://zhoujianan.com/assets/school/FDU_Course_ML/4.Auto%20Encoder/epoch.png)

![decoder](https://zhoujianan.com/assets/school/FDU_Course_ML/4.Auto%20Encoder/decoder.png)

# 5. Reproduction of ALBERT Model

## Task

With the application and development of pre-training model in natural language processing, machine reading comprehension no longer simply relies on the combination of network structure and word embedding. This paper briefly introduces the concepts of machine reading comprehension and pre-training language model, summarizes the research progress of machine reading comprehension based on ALBERT model, analyzes the performance of the current pre-training model on the relevant data set.

# Requirements

- python == 3.7
- pytorch == 1.0.1
- cuda version == 10.1

# Dataset

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

  > Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
  > A text file containing 5800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship. No more than 1 sentence has been extracted from any given news article. We have made a concerted effort to correctly associate with each sentence information about its provenance and any associated information about its author.

# Structure

![Structure](https://zhoujianan.com/assets/school/AIfinalproject/Structure.png)

# Results

| Model          | Parameters | SQuAD1.1  | SQuAD2.0  |
| -------------- | ---------- | --------- | --------- |
| ALBERT base    | 12M        | 89.3/82.1 | 79.1/76.1 |
| ALBERT large   | 18M        | 90.9/84.1 | 82.1/79.0 |
| ALBERT xlarge  | 59M        | 93.0/86.5 | 85.9/83.1 |
| ALBERT xxlarge | 233M       | 94.1/88.3 | 88.1/85.1 |

# Reference

- [BERT: Pre-training of deep bidirectional transformers for language understanding](https://arxiv.org/pdf/1810.04805.pdf)
- [Electra: Pre-training text encoders as discriminators rather than generators](https://arxiv.org/pdf/2003.10555.pdf)
- [ALBERT: A lite BERT for self-supervised learning of language representations](https://arxiv.org/pdf/1909.11942.pdf)
- [Know what you donâ€™t know: Unanswerable questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf)
- [NeurQuRI: Neural question requirement inspector for answerability prediction in machine reading comprehension](https://openreview.net/attachment?id=ryxgsCVYPr&name=original_pdf)
- [Attention-overattention neural networks for reading comprehension](https://arxiv.org/pdf/1607.04423.pdf)
- [Gated-attention readers for text comprehension](https://arxiv.org/pdf/1606.01549.pdf)
- [Text understanding with the attention sum reader network](https://arxiv.org/pdf/1603.01547.pdf)
- [XLNet: Generalized autoregressive pretraining for language understanding](http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)
