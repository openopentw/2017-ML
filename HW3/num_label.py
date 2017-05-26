#! python3
"""
@author: b04902053
"""

# import
import pandas
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

# Argvs
TRAIN = './data/train.csv'
OUTPUT = './con_submission.csv'

# load data
train_data = pandas.read_csv(TRAIN, encoding='big5').values[:,0]
train = train_data[:train_data.shape[0] - 1400]
vali  = train_data[train_data.shape[0] - 1400:]

"""
output_data = pandas.read_csv(OUTPUT, encoding='big5')
output_data = output_data.values
output = output_data[:,1]
"""

print('train')
label = np.zeros(7, dtype=int)
for i in range(7):
    label[i] = np.sum(train == i)
    print(label[i])

print('vali')
label = np.zeros(7, dtype=int)
for i in range(7):
    label[i] = np.sum(vali == i)
    print(label[i])
