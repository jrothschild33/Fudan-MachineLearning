# -*- coding: gbk -*-
# author: 周嘉楠

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
from myFunctions import readMyFile, myStandardize, extractFeaLab, myLinearReg

train, test = readMyFile()
new_train,new_test = myStandardize(train),myStandardize(test)
train_X_, train_Y_, test_X_ = extractFeaLab(new_train,train,new_test)
myLinearReg(train_X_, train_Y_, test_X_)