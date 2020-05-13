'''
Homework3:Sentiment Classification
Author:周嘉楠 19210980081
Model:RNN
'''

# =======================================================================
# 定义一些常用的函数
# =======================================================================
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

def load_training_data(path='/Data/laptop-train.txt'):
    with open(path, 'r',encoding='utf-8') as f:
        lines_ = f.readlines()
        lines = []
        # 删去包含conflict的行
        for line in lines_:
            if 'conflict' in line:
                pass
            else:
                lines.append(line)
        lines = [line.strip('\n').split('\t') for line in lines]
        x = [line[1].split(' ') for line in lines]
        y = [line[-1] for line in lines]
        fea_replace = {'positive':1,'negative':-1,'neutral':0}
        y = [fea_replace[i] if i in fea_replace else i for i in y]
    return x, y

def load_testing_data(path='/Data/laptop-test.txt'):
    with open(path, 'r',encoding='utf-8') as f:
        lines_ = f.readlines()
        lines = []
        # 删去包含conflict的行
        for line in lines_:
            if 'conflict' in line:
                pass
            else:
                lines.append(line)
        lines = [line.strip('\n').split('\t') for line in lines]
        X = [line[1].split(' ') for line in lines]
        Y = [line[-1] for line in lines]
        fea_replace = {'positive':1,'negative':-1,'neutral':0}
        Y = [fea_replace[i] if i in fea_replace else i for i in Y]
    return X,Y

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>0.5] = -1   # 大于0.5为negative
    outputs[outputs==0.5] = 0   # 等于0.5为neutral
    outputs[outputs<0.5] = 1    # 小于0.5为positive
    # eq:统计相等个数，sum:加总，item:将tensor转换成scalar
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct

# =======================================================================
# Train Word to Vector：用来训练 word to vector 的 word embedding
# =======================================================================

import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

path_prefix = './'


def train_word2vec(x):
    # 训练word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=250, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


if __name__ == "__main__":
    print("loading training data ...")
    train_x, y = load_training_data('./Data/laptop-train.txt')

    print("loading testing data ...")
    test_x, test_y = load_testing_data('./Data/laptop-test.txt')

    model = train_word2vec(train_x + test_x)

    print("saving model ...")
    # model.save(os.path.join(path_prefix, 'model/w2v_all.model'))
    model.save(os.path.join(path_prefix, './Model/w2v_all.model'))

# =======================================================================
# Data Preprocess：用来做data的预处理
# =======================================================================

from torch import nn
from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, sentences, sen_len, w2v_path="./Model/w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []
    def get_w2v_model(self):
        # 把之前训练好的word to vec 模型读进来
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size
    def add_embedding(self, word):
        # 把word加进embedding，并赋予他一个随机生成的representation vector
        # word只会是"<PAD>"或"<UNK>"
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)
    def make_embedding(self, load=True):
        print("Get embedding ...")
        # 取得训练好的 Word2vec word embedding
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError
        # 制作一个 word2idx 的 dictionary
        # 制作一个 idx2word 的 list
        # 制作一个 word2vector 的 list
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            #e.g. self.word2index['鲁'] = 1
            #e.g. self.index2word[1] = '鲁'
            #e.g. self.vectors[1] = '鲁' vector
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        print('')
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # 将"<PAD>"跟"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix
    def pad_sequence(self, sentence):
        # 将每个句子变成一样的长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence
    def sentence_word2idx(self):
        # 把句子里面的字转成相对应的index
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])
            # 将每个句子变成一样的长度
            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)
    def labels_to_tensor(self, y):
        # 把labels转成tensor
        y = [int(label) for label in y]
        return torch.LongTensor(y)

# =======================================================================
# Dataset
# =======================================================================

import torch
from torch.utils import data

class TwitterDataset(data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y
    def __getitem__(self, idx):
        if self.label is None: return self.data[idx]
        return self.data[idx], self.label[idx]
    def __len__(self):
        return len(self.data)


# =======================================================================
# Model：拿來训练的模型
# =======================================================================

import torch
from torch import nn
class LSTM_Net(nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, fix_embedding=True):
        super(LSTM_Net, self).__init__()
        # 制作 embedding layer
        self.embedding = torch.nn.Embedding(embedding.size(0),embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        # 是否将 embedding fix住，如果fix_embedding为False，在训练过程，embedding也会跟着被训练
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Sequential( nn.Dropout(dropout),
                                         nn.Linear(hidden_dim, 1),
                                         nn.Sigmoid() )
    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.lstm(inputs, None)
        # x 的 dimension (batch, seq_len, hidden_size)
        # 取用 LSTM 最后一层的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)
        return x


# =======================================================================
# Train：用来训练模型
# =======================================================================


import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    # 将model的模式设为train，这样optimizer就可以更新model的参数
    model.train()
    # 定义损失函数，这裡我们使用binary cross entropy loss
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    # 将模型的参数给optimizer，并给予适当的learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_acc = 0, 0, 0
    for epoch in range(n_epoch):
        total_loss, total_acc = 0, 0
        # 这段做training
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device, dtype=torch.long) # device为"cuda"，将inputs转成torch.cuda.LongTensor
            labels = labels.to(device, dtype=torch.float) # device为"cuda"，将labels转成torch.cuda.FloatTensor，因为等等要喂进criterion，所以型态要是float
            optimizer.zero_grad() # 由于loss.backward()的gradient会累加，所以每次喂完一个batch后需要归零
            outputs = model(inputs) # 将input喂给模型
            outputs = outputs.squeeze() # 去掉最外面的dimension，好让outputs可以喂进criterion()
            loss = criterion(outputs, labels) # 计算此时模型的training loss
            loss.backward() # 算loss的gradient
            optimizer.step() # 更新训练模型的参数
            correct = evaluation(outputs, labels) # 计算此时模型的training accuracy
            total_acc += (correct / batch_size)
            total_loss += loss.item()
            print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            	epoch+1, i+1, t_batch, loss.item(), correct*100/batch_size), end='\r')
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss/t_batch, total_acc/t_batch*100))

        # 这段做validation
        model.eval() # 将model的模式设为eval，这样model的参数就会固定住
        with torch.no_grad():
            total_loss, total_acc = 0, 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device, dtype=torch.long) # device为"cuda"，将inputs转成torch.cuda.LongTensor
                labels = labels.to(device, dtype=torch.float) # device为"cuda"，将labels转成torch.cuda.FloatTensor，因为等等要喂进criterion，所以型态要是float
                outputs = model(inputs) # 将input喂给模型
                outputs = outputs.squeeze() # 去掉最外面的dimension，好让outputs可以喂进criterion()
                loss = criterion(outputs, labels) # 计算此时模型的validation loss
                correct = evaluation(outputs, labels) # 计算此时模型的validation accuracy
                total_acc += (correct / batch_size)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss/v_batch, total_acc/v_batch*100))
            if total_acc > best_acc:
                # 如果validation的结果优于之前所有的结果，就把当下的模型存下来以备之后做预测时使用
                best_acc = total_acc
                #torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, "{}/Model/ckpt.model".format(model_dir))
                print('saving model with acc {:.3f}'.format(total_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train() # 将model的模式设为train，这样optimizer就可以更新model的参数（因为刚刚转成eval模式）


# =======================================================================
# Test：用来对laptop-test.txt做预测
# =======================================================================

import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F


def testing(batch_size, test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for i, inputs in enumerate(test_loader):
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            outputs[outputs > 0.5] = -1  # 大于0.5为negative
            outputs[outputs == 0.5] = 0  # 等于0.5为neutral
            outputs[outputs < 0.5] = 1  # 小于0.5为positive
            ret_output += outputs.int().tolist()

    return ret_output


# =======================================================================
# Main
# =======================================================================

import os
import torch
import argparse
import numpy as np
from torch import nn
from gensim.models import word2vec
from sklearn.model_selection import train_test_split

# 通过torch.cuda.is_available()的回传值进行判断是否有使用GPU的环境，如果有的话device就设为"cuda"，没有的话就设为"cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 处理好各个data的路径
train_with_label = os.path.join(path_prefix, './Data/laptop-train.txt')
testing_data = os.path.join(path_prefix, './Data/laptop-test.txt')

w2v_path = os.path.join(path_prefix, './Model/w2v_all.model') # 处理word to vec model的路径

# 定义句子长度、要不要固定embedding、batch大小、要训练几个epoch、learning rate的值、model的资料夹路径
sen_len = 30
fix_embedding = True # fix embedding during training
batch_size = 128
epoch = 10
lr = 0.001
# model_dir = os.path.join(path_prefix, 'model/') # model directory for checkpoint model
model_dir = path_prefix # model directory for checkpoint model

print("loading data ...") # 把'laptop-train.txt'读进来
train_x, y = load_training_data(train_with_label)


# 对input跟labels做预处理
preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

# 製作一个model的对象
model = LSTM_Net(embedding, embedding_dim=250, hidden_dim=250, num_layers=1, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device) # device为"cuda"，model使用GPU来训练(喂进去的inputs也需要是cuda tensor)

# 把data分为training data跟validation data(将一部份training data拿去当作validation data)
X_train, X_val, y_train, y_val = train_x[:1620], train_x[1620:], y[:1620], y[1620:]

# 把data做成dataset供dataloader取用
train_dataset = TwitterDataset(X=X_train, y=y_train)
val_dataset = TwitterDataset(X=X_val, y=y_val)

# 把data 转成 batch of tensors
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                            batch_size = batch_size,
                                            shuffle = True,
                                            #num_workers = 2
                                           )

val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            #num_workers = 2
                                         )

# 开始训练
training(batch_size, epoch, lr, model_dir, train_loader, val_loader, model, device)

# =======================================================================
# Predict and Write to txt file
# =======================================================================

# 开始测试模型并做预测
print("loading testing data ...")
test_x,test_y = load_testing_data(testing_data)
preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()
test_dataset = TwitterDataset(X=test_x, y=None)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            #num_workers = 2
                                          )
print('\nload model ...')
model = torch.load(os.path.join(model_dir, './Model/ckpt.model'))
outputs_ = testing(batch_size, test_loader, model, device)

# 将数字标签转换回字符串
fea_replace = {'positive':1,'negative':-1,'neutral':0}
fea_replace = {value:key for key,value in fea_replace.items()}
outputs = [fea_replace[i] if i in fea_replace else i for i in outputs_]

# 将文件转换为CSV
tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": outputs})
print("save file ...")
tmp.to_csv(os.path.join(path_prefix, './Result/result_rnn.txt'), sep='\t', index=False)
print("Finish Predicting")

# =======================================================================
# 对模型进行评估
# =======================================================================


import sys
import codecs
from typing import List
from sklearn import metrics

def calc_acc_f1(golds: List[str], predicts: List[str]) -> tuple:
    return metrics.accuracy_score(golds, predicts), \
           metrics.f1_score(golds, predicts, labels=['positive', 'negative', 'neutral'], average='macro')

# 将测试集数字标签转换回字符串
fea_replace = {'positive': 1, 'negative': -1, 'neutral': 0}
fea_replace = {value: key for key, value in fea_replace.items()}
predicts = [fea_replace[i] if i in fea_replace else i for i in test_y]
golds = outputs

# 计算F1-Score
acc, f1_score = calc_acc_f1(golds, predicts)
print("acc: {}, f1: {}".format(acc, f1_score))

# 画出混淆矩阵
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']

# 输出分类信息
print('分类信息：\n',metrics.classification_report(golds, predicts))

# 输出混淆矩阵信息
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(golds, predicts, labels=['positive', 'negative', 'neutral'])
sns_heatmap = sns.heatmap(C2,annot=True,ax=ax)
ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')
fig = sns_heatmap.get_figure()
fig.savefig('./Result/RNN_confusion_matrix.png',dpi = 300)
