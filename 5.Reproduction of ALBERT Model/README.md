# Description

With the application and development of pre-training model in natural language processing, machine reading comprehension no longer simply relies on the combination of network structure and word embedding. This paper briefly introduces the concepts of machine reading comprehension and pre-training language model, summarizes the research progress of machine reading comprehension based on ALBERT model, analyzes the performance of the current pre-training model on the relevant data set.

[Click here to download the report](https://zhoujianan.com/assets/school/AIfinalproject/Report.pdf)

# Requirements

- python == 3.7
- pytorch == 1.0.1
- cuda version == 10.1

# Dataset

- [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/)

  > Stanford Question Answering Dataset (SQuAD) is a reading comprehension dataset, consisting of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or span, from the corresponding reading passage, or the question might be unanswerable.

- [MRPC](https://www.microsoft.com/en-us/download/details.aspx?id=52398)
  > A text file containing 5800 pairs of sentences which have been extracted from news sources on the web, along with human annotations indicating whether each pair captures a paraphrase/semantic equivalence relationship. No more than 1 sentence has been extracted from any given news article. We have made a concerted effort to correctly associate with each sentence information about its provenance and any associated information about its author.

# Code

![Structure](https://zhoujianan.com/assets/school/AIfinalproject/Structure.png)

## Data Preprocess

```
$ download_mrpc.py --- auto-download mrpc dataset

$ link_test.py --- test if the downloading link is still available

$ mrpc.py --- mrpc processor; consists of MNLI processor, COLA processor, STS-B processor, STS-2 processor, QQP processor, QNLI processor, RTE processor, WNLI processor

$ squad.py --- squad processor; consists of SQuAD processor, SQuADV1Processor, SQuADV2Processor, SQuADresult

$ utils.py --- base class for sequence classification model, consists of InputFeatures, DataProcessor, SingleSentenceClassificationProcessor
```

## Model Construction

```
$ configuration.py --- albert configuration; consists of  PretrainedConfig, AlbertConfig

$ tokenization.py --- albert tokenization; PreTrainedTokenizer, AlbertTokenizer

$ convert.py --- convert albert tf checkpoint to pytorch

$ model.py --- build albert model; consists of AlbertEmbeddings, AlbertAttention, AlbertLayer, AlbertLayerGroup, AlbertTransformer, AlbertPreTrainedModel, AlbertModel, AlbertForSequenceClassification

$ optimization.py --- optimize albert model, consists of AdamW optimizer

$ pipeline.py --- pipelines for implementing sketchy reading and intensive reading, consists of PipelineDataFormat, JsonPipelineDataFormat, Pipeline, TextClassificationPipeline, NerPipeline
```

## Model Evaluation

```
$ metrics.py --- outputs measured by F1-score and exact match, consists of compute_predictions_logits_av, compute_predictions_log_probs
```

## Model Execution

```
$ download.py --- download command

$ convert.py --- convert command

$ server.py --- serving server command

$ train.py --- train command

$ run.py --- run command
```

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
- [Know what you don’t know: Unanswerable questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf)
- [NeurQuRI: Neural question requirement inspector for answerability prediction in machine reading comprehension](https://openreview.net/attachment?id=ryxgsCVYPr&name=original_pdf)
- [Attention-overattention neural networks for reading comprehension](https://arxiv.org/pdf/1607.04423.pdf)
- [Gated-attention readers for text comprehension](https://arxiv.org/pdf/1606.01549.pdf)
- [Text understanding with the attention sum reader network](https://arxiv.org/pdf/1603.01547.pdf)
- [XLNet: Generalized autoregressive pretraining for language understanding](http://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf)
