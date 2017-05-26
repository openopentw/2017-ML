#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

# imports# {{{
import pandas
import numpy as np# }}}

"""
Input Data
format: DATA['month']['feature']['hour']
"""
# input data
DATA = pandas.read_csv('data/train.csv', encoding='big5')
DATA = DATA.values

# append the same features together
DATA = DATA[ np.argsort(DATA[:,2], kind='mergesort') ]
DATA = DATA[:,3:]
DATA = DATA.reshape(18, int(DATA.size / 18))

# convert DATA to float
DATA[DATA == 'NR'] = '0'
DATA = DATA.astype(float)

# normalize DATA
MEAN = np.mean(DATA, axis=1).reshape(18, 1)
VAR = np.var(DATA, axis=1).reshape(18, 1)

DATA = (DATA - MEAN) / VAR

"""
# save to 'sort_train.csv'
f = open("sort_train.csv", "w+")
for i in range(DATA.shape[0]):
    for j in range(DATA.shape[1]):
        if j > 0:
            print(',', file=f, end='')
        print(DATA[i][j], file=f, end='')
    print('\n', file=f, end='')
f.close()
"""






FEATURE = np.array([8, 9])
N_FEATURE = FEATURE.size





DIM = 2

DATA_DIM = np.zeros((DIM, DATA.shape[0], DATA.shape[1]))
for i in range(DIM):
    DATA_DIM[i] = DATA ** (i+1)

DATA = np.zeros(12 * 471 * (N_FEATURE*9*DIM+1))
pos = 0
for i in range(12):
    for j in range(i*471, (i+1)*471):
        for k in range(N_FEATURE):
            for l in range(DIM):
                DATA[pos:pos+9] = DATA_DIM[l][FEATURE[k]][j:j+9]
                pos += 9
        DATA[pos] = DATA_DIM[0][9][j+9]
        pos += 1
DATA = DATA.reshape(12*471, N_FEATURE*9*DIM + 1)
