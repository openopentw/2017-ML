#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

import pandas
import numpy as np

# input data #
DATA = pandas.read_csv('data/train.csv', encoding='big5')
DATA = DATA.values
DATA = DATA[:,3:13]
DATA[DATA == 'NR'] = '0'
DATA = DATA.astype(float)

# turn 2D data to 3D data #
NDATE = int(DATA.shape[0] / 18)
DATA = DATA.reshape(NDATE, 18, DATA.shape[1])
DATA[:,:,9] = DATA[:,9,9].reshape(DATA.shape[0], 1)  # let DATA[:,:,9] be the solution

# slice half of DATA to TRAIN & TEST #
HALF = int(DATA.shape[0] / 2)
TRAIN = DATA[:HALF]
TEST = DATA[HALF:]

"""
Start training
"""

YETA = 1e-9
ROOT_SUM_b = 1e-9
ROOT_SUM_W = 1e-9
def learning_rate(grad_b, grad_W, train, b, it):
    yeta = YETA

    global ROOT_SUM_b
    ROOT_SUM_b += grad_b ** 2
    sigma_b = np.sqrt(ROOT_SUM_b)

    global ROOT_SUM_W
    ROOT_SUM_W += grad_W ** 2
    sigma_W = np.sqrt(ROOT_SUM_W)

    return [yeta/sigma_b, yeta/sigma_W]

def gradient_descent(X, b, W, it):
    train = X[:,:,:9]
    ans = X[:,:,9]
    loss = np.sum(ans - (b + np.sum(W * train, axis=2)), axis=1)
    grad_b = 2 * np.sum(loss)
    grad_W = 2 * np.sum(loss.reshape(loss.size,1,1) * train, axis=0)
    [rate_b, rate_W] = learning_rate(grad_b, grad_W, train, b, it)
    new_b = b + rate_b * grad_b
    new_W = W + rate_W * grad_W
    return [new_b, new_W]

def calc_loss(b, W):
    rand_num = (np.random.random(20) * (HALF - 1)).astype(int)
    X = TRAIN[rand_num][:,2:10]
    train = X[:,:,:9]
    ans = X[:,:,9]
    loss = np.sum(ans - (b + np.sum(W * train, axis=2)), axis=1)
    return np.sum(loss ** 2)

b = 0
W = np.zeros([8,9])
# W[7] = np.ones(9) / 9
LOSS = 0
LAST_LOSS = 0

OUT = open('out', 'w+')

for i in range(2000):
# for i in range(1):
    train_num = []
    if i < 100:
        train_num = (np.random.random(10) * (HALF - 1)).astype(int)
    else:
        train_num = np.arange(HALF)
    X = TRAIN[train_num][:,2:10]
    [b, W] = gradient_descent(X, b, W, i + 1)
    LAST_LOSS = LOSS
    LOSS = calc_loss(b, W)
    ERROR = (LAST_LOSS - LOSS) / LOSS
    # print(LOSS, ERROR)
    # print(LOSS, file=OUT)
    print(LOSS)

print('=== END ===')
# print(b)
# print(W)
