#%%

'''
Homework3:Sentiment Classification
Author:周嘉楠 19210980081
Model:CNN
'''

# =======================================================================
# 定义一些常用的函数
# =======================================================================
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F


def load_training_data(path='./Data/laptop-train.txt'):
    with open(path, 'r', encoding='utf-8') as f:
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
        fea_replace = {'positive': 1, 'negative': -1, 'neutral': 0}
        y = [fea_replace[i] if i in fea_replace else i for i in y]
    return x, y


def load_testing_data(path='./Data/laptop-test.txt'):
    with open(path, 'r', encoding='utf-8') as f:
        lines_ = f.readlines()
        lines = []
        # 删去包含conflict的行
        for line in lines_:
            if 'conflict' in line:
                pass
            else:
                lines.append(line)
        lines = [line.strip('\n').split('\t') for line in lines]
        X_split = [line[1].split(' ') for line in lines]
        for i in X_split:
            if len(i) < 5:
                i.append('. . . . .')
        X = [' '.join(i) for i in X_split]
        Y = [line[-1] for line in lines]
        fea_replace = {'positive': 1, 'negative': -1, 'neutral': 0}
        Y = [fea_replace[i] if i in fea_replace else i for i in Y]
    return X,Y

# =======================================================================
# 定义CNN模型
# =======================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN_Text(nn.Module):

    def __init__(self, args):
        super(CNN_Text, self).__init__()
        self.args = args

        V = args.embed_num
        D = args.embed_dim
        C = args.class_num
        Ci = 1
        Co = args.kernel_num
        Ks = args.kernel_sizes

        self.embed = nn.Embedding(V, D)
        # self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, D)) for K in Ks])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        self.dropout = nn.Dropout(args.dropout)
        self.fc1 = nn.Linear(len(Ks) * Co, C)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.embed(x)  # (N, W, D)

        if self.args.static:
            x = Variable(x)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit


# =======================================================================
# Training
# =======================================================================
import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, target = batch.text, batch.label
            feature = feature.data.t()  # x.t() x是不变的，所以重新赋值
            target = target.data.sub(1)  # x.sub() x是不变的，所以重新赋值
            if args.cuda:
                feature, target = feature.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\rBatch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(steps,
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    corrects, avg_loss = 0, 0
    for batch in data_iter:
        feature, target = batch.text, batch.label
        feature = feature.data.t()  # x.t() x是不变的，所以重新赋值
        target = target.data.sub(1)  # x.sub() x是不变的，所以重新赋值
        if args.cuda:
            feature, target = feature.cuda(), target.cuda()

        logit = model(feature)
        loss = F.cross_entropy(logit, target, size_average=False)

        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_steps_{}.pt'.format(save_prefix, steps)
    torch.save(model.state_dict(), save_path)

# =======================================================================
# 辅助训练语料库
# =======================================================================
import re
import os
import random
import tarfile
import urllib
from torchtext import data


class TarDataset(data.Dataset):
    @classmethod
    def download_or_unzip(cls, root):
        path = os.path.join(root, cls.dirname)
        if not os.path.isdir(path):
            tpath = os.path.join(root, cls.filename)
            if not os.path.isfile(tpath):
                print('downloading')
                urllib.request.urlretrieve(cls.url, tpath)
            with tarfile.open(tpath, 'r') as tfile:
                print('extracting')
                def is_within_directory(directory, target):
                    
                    abs_directory = os.path.abspath(directory)
                    abs_target = os.path.abspath(target)
                
                    prefix = os.path.commonprefix([abs_directory, abs_target])
                    
                    return prefix == abs_directory
                
                def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
                
                    for member in tar.getmembers():
                        member_path = os.path.join(path, member.name)
                        if not is_within_directory(path, member_path):
                            raise Exception("Attempted Path Traversal in Tar File")
                
                    tar.extractall(path, members, numeric_owner=numeric_owner) 
                    
                
                safe_extract(tfile, root)
        return os.path.join(path, '')


class MR(TarDataset):

    url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
    filename = 'rt-polaritydata.tar'
    dirname = './Model/rt-polaritydata'

    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, text_field, label_field, path=None, examples=None, **kwargs):
        def clean_str(string):
            string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
            string = re.sub(r"\'s", " \'s", string)
            string = re.sub(r"\'ve", " \'ve", string)
            string = re.sub(r"n\'t", " n\'t", string)
            string = re.sub(r"\'re", " \'re", string)
            string = re.sub(r"\'d", " \'d", string)
            string = re.sub(r"\'ll", " \'ll", string)
            string = re.sub(r",", " , ", string)
            string = re.sub(r"!", " ! ", string)
            string = re.sub(r"\(", " \( ", string)
            string = re.sub(r"\)", " \) ", string)
            string = re.sub(r"\?", " \? ", string)
            string = re.sub(r"\s{2,}", " ", string)
            return string.strip()

        text_field.preprocessing = data.Pipeline(clean_str)
        fields = [('text', text_field), ('label', label_field)]

        if examples is None:
            path = self.dirname if path is None else path
            examples = []
            with open(os.path.join(path, 'rt-polarity.neg'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'negative'], fields) for line in f]
            with open(os.path.join(path, 'rt-polarity.pos'), errors='ignore') as f:
                examples += [
                    data.Example.fromlist([line, 'positive'], fields) for line in f]
        super(MR, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, dev_ratio=.1, shuffle=True, root='.', **kwargs):
        path = cls.download_or_unzip(root)
        examples = cls(text_field, label_field, path=path, **kwargs).examples
        if shuffle: random.shuffle(examples)
        dev_index = -1 * int(dev_ratio*len(examples))

        return (cls(text_field, label_field, examples=examples[:dev_index]),
                cls(text_field, label_field, examples=examples[dev_index:]))

# =======================================================================
# Main
# =======================================================================

import os
import argparse
import datetime
import torch
import torchtext.data as data
import torchtext.datasets as datasets

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval', type=int, default=1,
                    help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100,
                    help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000,
                    help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5',
                    help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_known_args()[0]


# load SST dataset
def sst(text_field, label_field, **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train_data, dev_data, test_data),
        batch_sizes=(args.batch_size,
                     len(dev_data),
                     len(test_data)),
        **kargs)
    return train_iter, dev_iter, test_iter


# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    train_iter, dev_iter = data.Iterator.splits(
        (train_data, dev_data),
        batch_sizes=(args.batch_size, len(dev_data)),
        **kargs)
    return train_iter, dev_iter


# load data
print("\nLoading data...")
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)
# train_iter, dev_iter, test_iter = sst(text_field, label_field, device=-1, repeat=False)


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available();
del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = CNN_Text(args)
args.snapshot = './Model/snapshot/2020-05-09_16-18-46/best_steps_8700.pt'
# 训练模式：
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()


# # train or predict
# if args.predict is not None:
#     label = predict(args.predict, cnn, text_field, label_field, args.cuda)
#     print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
# elif args.test:
#     try:
#         eval(test_iter, cnn, args)
#     except Exception as e:
#         print("\nSorry. The test dataset doesn't exist.\n")
# else:
#     print()
#     try:
#         train(train_iter, dev_iter, cnn, args)
#     except KeyboardInterrupt:
#         print('\n' + '-' * 89)
#         print('Exiting from training early')


# 开始测试模型并做预测

print("loading testing data ...")
path_prefix = './'
testing_data = os.path.join(path_prefix, './Data/laptop-test.txt')
test_x,test_y = load_testing_data(testing_data)


mylabel = []
for i in test_x:
    args.predict = i
    label = predict(args.predict, cnn, text_field, label_field, args.cuda)
    mylabel.append(label)
    # print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))


# 将文件转换为CSV
tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": mylabel})
print("save file ...")
tmp.to_csv(os.path.join(path_prefix, './Result/result_cnn.txt'), sep='\t', index=False)
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
golds = [fea_replace[i] if i in fea_replace else i for i in test_y]
predicts = mylabel

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
# label = ['positive', 'negative', 'neutral']
sns.set()
f,ax=plt.subplots()
C2= confusion_matrix(golds, predicts, labels=['positive', 'negative', 'neutral'])
sns_heatmap = sns.heatmap(C2,annot=True,ax=ax)
ax.set_title('confusion matrix')
ax.set_xlabel('predict')
ax.set_ylabel('true')
fig = sns_heatmap.get_figure()
fig.savefig('./Result/CNN_confusion_matrix.png',dpi = 300)