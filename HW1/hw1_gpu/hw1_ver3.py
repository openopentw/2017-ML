#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

# imports# {{{
import pandas
import numpy as np
from numbapro import vectorize
@vectorize(["float64(float64, float64)"], target='gpu')
# }}}

"""
Input Data
format: DATA['month']['feature']['hour']
"""
# {{{
# input data
DATA = pandas.read_csv('data/train.csv', encoding='big5')
DATA = DATA.values

# append the same features together
DATA = DATA.reshape(12, 20 * 18,  int(DATA.size / 12 / (20*18)))
for i in range(12):
    DATA[i] = DATA[i][ np.argsort(DATA[i][:,2], kind='mergesort') ]
DATA = DATA[:,:,3:]
DATA = DATA.reshape(12, 18, int(DATA.size / 12 / 18))

# convert DATA to float
DATA[DATA == 'NR'] = '0'
DATA = DATA.astype(float)
# }}}

"""
Adjust Data To Be Trainable
format: DATA[ i-th data to train (month * feature * hour) ]
"""
DIM = 2
FEATURE = np.array([8, 9])
N_FEATURE = FEATURE.size
# {{{
# let DATA be "0123456789 12345678910" format
OLD_DATA = DATA

DATA = np.zeros(12 * 471 * (N_FEATURE*9*DIM+1))
pos = 0
for i in range(12):
    for j in range(471):
        for l in range(DIM):
            for k in range(N_FEATURE):
                DATA[pos:pos+9] = OLD_DATA[i][FEATURE[k]][j:j+9] ** (l+1)
                pos += 9
        DATA[pos] = OLD_DATA[i][9][j+9]
        pos += 1
DATA = DATA.reshape(12*471, N_FEATURE*9*DIM + 1)
# }}}

"""
Start training
"""
LMBD = 1e4

YETA = 1
ROOT_SUM_b = 0
ROOT_SUM_W = np.zeros((1, N_FEATURE*9*DIM))

b = 0
W = np.zeros((1, N_FEATURE*9*DIM))
LOSS = 0
LAST_LOSS = 0

"""
training for-loop
"""

train = DATA[ : int(DATA.shape[0]/2)]
test = DATA[int(DATA.shape[0]/2) : ]
# train = DATA[:]
# test = DATA[:]
num_train = train.shape[0]
num_test = test.shape[0]

X = train[:,:-1]
ans = train[:,-1].reshape((train.shape[0], 1))
testX = test[:,:-1]
testans = test[:,-1].reshape((test.shape[0], 1))

for i in range(500):
    # gradient
    sqrt_loss = ans - ( b + np.dot(X, W.T ) )
    grad_b = 2 * np.sum(sqrt_loss)
    grad_W = 2 * (np.sum(sqrt_loss * X, axis=0) + LMBD * W)

    # learning_rate
    ROOT_SUM_b += grad_b ** 2
    rate_b = YETA / np.sqrt(ROOT_SUM_b)
    ROOT_SUM_W += grad_W ** 2
    rate_W = YETA / np.sqrt(ROOT_SUM_W)

    # descent
    b += rate_b * grad_b
    W += rate_W * grad_W

    # loss
    LAST_LOSS = LOSS
    sqrt_loss = testans - ( b + np.dot(testX, W.T ) )
    LOSS = np.sum(sqrt_loss ** 2) + LMBD * np.sum(W ** 2)

    print(i, np.sqrt(LOSS/num_test))
    if i > 6000 and LOSS > LAST_LOSS:
        break

print(b)
print(W)

"""
Calculate Ans & Print Out
"""
test = np.genfromtxt('data/test_X.csv', delimiter=',')

# divide test_data into several ids
test = test[:,2:]
test = test.reshape(int(test.size / 18 / 9), 18, 9)

# convert test to float
test = np.nan_to_num(test)
test = test.astype(float)

# delete unused test_data
test = test[:, FEATURE,:]
test = test.reshape(( test.shape[0], int(test.size / test.shape[0]) ))

# adjust test_data to testable
dim_test = np.zeros(( test.shape[0], int(test.size/test.shape[0])*DIM ))
for i in range(test.shape[0]):
    pos = 0
    for j in range(DIM):
        dim_test[i][pos:pos+9*N_FEATURE] = test[i][:] ** (j+1)
        pos += 9*N_FEATURE

ans = b + np.dot(dim_test, W.T)

# save to 'submission.csv'
f = open("submission.csv", "w+")
print("id,value", file = f, end = "\n")
for i in range(ans.size):
    print("id_" + str(i) + "," + str(np.round(ans[i][0])), file = f, end = "\n")
f.close()
