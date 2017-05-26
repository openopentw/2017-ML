#! python3
"""
@author: b04902053
"""

# import# {{{
import pandas
import numpy as np
import sys
# }}}

# Argvs# {{{
# TRAIN_FILE = sys.argv[1]
# TEST_FILE = sys.argv[2]
# OUTPUT = sys.argv[3]
TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
OUTPUT = "submission.csv"# }}}

def load_data():# {{{
    train_data = pandas.read_csv(TRAIN_FILE, encoding='big5')
    train_data = train_data.values

    x_train_data = np.zeros((train_data.shape[0], 2304), int)
    for i in range(train_data.shape[0]):
        x_train_data[i,:] = np.fromstring(train_data[i,1], dtype=np.int, sep=' ')
    x_train = x_train_data.reshape(x_train_data.shape[0], 48, 48)

    test_data = pandas.read_csv(TEST_FILE, encoding='big5')
    test_data = test_data.values

    x_test_data = np.zeros((test_data.shape[0], 2304), int)
    for i in range(test_data.shape[0]):
        x_test_data[i,:] = np.fromstring(test_data[i,1], dtype=np.int, sep=' ')
    x_test = x_test_data.reshape(x_test_data.shape[0], 48, 48)

    return (x_train, x_test)# }}}

# TODO: manually change the following value
LOADED = 0
if LOADED == 0:
    (x_train, x_test) = load_data()

# for i in range(x_train.shape[0]):
    # imsave('pngs/train/'+str(i)+'.png', x_train[i])

# for i in range(x_test.shape[0]):
    # imsave('pngs/test/'+str(i)+'.png', x_test[i])
