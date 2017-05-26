#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

import pandas
import numpy as np

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

# slice half of DATA to TRAIN & TEST
TRAIN = DATA[:6,:,:]
TEST = DATA[6:,:,:]
# }}}

"""
Start training
"""

def gradient(X, b, W):# {{{
    grad_b = 0
    grad_W = np.zeros(9)
    it_range = range(X.size - 9)
    for i in it_range:
        train = X[i:i+9]
        ans = X[i+9]
        sqrt_loss = ans - (b + np.sum(W * train))
        grad_b += 2 * sqrt_loss
        grad_W += 2 * sqrt_loss * train
    return [grad_b, grad_W]# }}}

YETA = 1
ROOT_SUM_b = 0
ROOT_SUM_W = np.zeros(9)
def learning_rate(grad_b, grad_W):# {{{
    yeta = YETA

    global ROOT_SUM_b
    ROOT_SUM_b += grad_b ** 2
    sigma_b = np.sqrt(ROOT_SUM_b)

    global ROOT_SUM_W
    ROOT_SUM_W += grad_W ** 2
    sigma_W = np.sqrt(ROOT_SUM_W)

    return [yeta/sigma_b, yeta/sigma_W]# }}}

def gradient_descent(b, W, grad_b, grad_W):# {{{
    [rate_b, rate_W] = learning_rate(grad_b, grad_W)
    new_b = b + rate_b * grad_b
    new_W = W + rate_W * grad_W
    return [new_b, new_W]# }}}

def calc_loss(b, W):# {{{
    # rand_month = (np.random.random(20) * (6)).astype(int)
    # rand_hour = (np.random.random(20) * (480 - 10)).astype(int)
    loss = 0
    for i in range(6):
        it_range = range(TEST.shape[2] - 10)
        for j in it_range:
            X = TEST[i, 9, j:j+10]
            test = X[:9]
            ans = X[9]
            loss += (ans - (b + np.sum(W * test))) ** 2
    return loss# }}}

b = 0
W = np.zeros(9)
LOSS = 0
LAST_LOSS = 0

for i in range(800000):
    it = i
    grad_b = 0
    grad_W = np.zeros(9)
    if it < 3000:
        rand_num = (np.random.random(20) * (TRAIN.shape[2] - 10)).astype(int)
        for j in range(6):
            for k in range(20):
                [return_grad_b, return_grad_W] = gradient(TRAIN[j][9][rand_num[k]:rand_num[k] + 10], b, W)
                grad_b += return_grad_b
                grad_W += return_grad_W
    else:
        for j in range(6):
            [return_grad_b, return_grad_W] = gradient(TRAIN[j][9], b, W)
            grad_b += return_grad_b
            grad_W += return_grad_W
    [b, W] = gradient_descent(b, W, grad_b, grad_W)

    LAST_LOSS = LOSS
    LOSS = calc_loss(b, W)

    grad = grad_b**2 + np.sum(grad_W**2)
    print(it, LOSS, grad)
    if it > 6000 and LOSS > LAST_LOSS:
        break
print(b)
print(W)

"""
Calculate Ans & Print Out
"""

def submission(b, W):# {{{
    # input test_data
    test = np.genfromtxt('data/test_X.csv', delimiter=',')

    # divide test_data into several ids
    test = test[:,2:]
    test = test.reshape(int(test.size / 18 / 9), 18, 9)

    # convert test to float
    test[test == 'NR'] = '0'
    test = test.astype(float)

    # delete unused test_data
    test = test[:,9,:]

    ans = b + np.dot(test, W.T)

    # print(ans.shape)
    # print(ans)

    # save to 'submission.csv'
    f = open("submission.csv", "w+")
    print("id,value", file = f, end = "\n")
    for i in range(ans.size):
        print("id_" + str(i) + "," + str(ans[i]), file = f, end = "\n")
    f.close()

    return# }}}

submission(b, W)
