#! python3
"""
Created on Thu Mar  9 09:31:27 2017
@author: b04902053
"""

import pandas
import numpy as np

b = 1.51175858535
W = np.array([ -3.73256568e-2, 6.91644518e-4, 1.67464809e-1, -1.94980980e-1 , -5.97590770e-2, 4.97724183e-1, -5.27222258e-1, -5.9309291e-2 , 1.14795086 ])

def submission(b, W):
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

    return
