# import# {{{
import pandas
import numpy as np
# import sys
# }}}
from scipy.misc import imsave

# Argvs# {{{
# TRAIN_FILE = sys.argv[1]
# TEST_FILE = sys.argv[2]
# OUTPUT = sys.argv[3]

TRAIN_FILE = "data/train.csv"
TEST_FILE = "data/test.csv"
VALI_FILE = "data/fer2013/fer_test.csv"
OUTPUT = "submission.csv"

BAD_TRAIN_FILE = 'data/bad_train_data.txt'
# BAD_TEST_FILE = 'data/bad_test_data.txt'  # for unsupervised use

NUM_CLASSES = 7
# }}}

def histeq(data):# {{{
    for i in range(data.shape[0]):
        imhist, bins = np.histogram(data[i], 256, normed=False)
        cdf = imhist.cumsum()
        cdf = 255 * cdf / cdf[-1]
        data[i] = np.interp(data[i], bins[:-1], cdf)
    return data
# }}}

def load_train_data(): # with execute# {{{
    # load from file
    train_data = pandas.read_csv(TRAIN_FILE, encoding='big5')
    train_data = train_data.values

    # remove bad data
    bad_train_data = np.genfromtxt(BAD_TRAIN_FILE, dtype=int)
    train_data = np.delete(train_data, bad_train_data, axis=0)

    # generate y_train_data# {{{
    y_train_data = train_data[:,0].reshape(train_data.shape[0], 1)
    # y_train = np_utils.to_categorical(y_train_data, 7)# }}}
    y_train = y_train_data

    # generate x_train_data# {{{
    # split train
    x_train_data = np.zeros((train_data.shape[0], 2304), int)
    for i in range(train_data.shape[0]):
        x_train_data[i,:] = np.fromstring(train_data[i,1], dtype=np.int, sep=' ')
    x_train_data = x_train_data.astype(float)
    # # normalize
    # x_train_data = normalize(x_train_data)
    # histeq
    x_train_data = histeq(x_train_data)
    # }}}

    return (x_train_data, y_train)
(x_train_data, y_train) = load_train_data()# }}}

# reshape# {{{
x_train = x_train_data.reshape(x_train_data.shape[0], 48, 48, 1)
x_train = x_train_data.reshape(x_train_data.shape[0], 48, 48)
# x_test  = x_test_data.reshape(x_test_data.shape[0], 48, 48, 1)
# x_vali  = x_vali_data.reshape(x_vali_data.shape[0], 48, 48, 1)
# }}}

for i in range(x_train.shape[0]):
    imsave('./pngs/train_histeq_nonorm/'+str(i)+'.png', x_train[i])
