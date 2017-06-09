"""
@author: b04902053
"""
use_device = 'gpu'  # cpu / gpu
# Use Device# {{{
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
if use_device == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif use_device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# }}}
# import# {{{
import pandas as pd
import numpy as np
import sys
# keras# {{{
import keras.backend as K
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Flatten, Embedding, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Dot, Concatenate
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}
# Parameter #
ID = 1
# SEED = 87
EPOCHS = 50
TRAIN_BATCH_SIZE = int(899873 / 32)
TRAIN_BATCH_SIZE = 1024
TRAIN_BATCH_SIZE = 100336
LOAD_BATCH_SIZE = 100336
# argv# {{{
print('ID = {}'.format(ID))
train_path  = '../data/train.csv'
test_path   = '../data/test.csv'
output_path = '../subm/train_vote_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))

weights_path = '../weights/vote_weights_{}.h5'.format(ID)
model_path   = '../model/vote_{}.h5'.format(ID)
print('Will save model to: {}'.format(model_path))
# }}}
# Load Data #
# load train data# {{{
train = pd.read_csv(train_path)[['UserID', 'MovieID', 'Rating']].values
train[:,0] = train[:,0] - 1
train[:,1] = train[:,1] - 1
user_size  = train[:,0].max() + 1
movie_size = train[:,1].max() + 1
# }}}
# shuffle train# {{{
# np.random.seed(SEED)
# indices = np.arange(train.shape[0])
# np.random.shuffle(indices)
# train = train[indices]
# }}}
# load test data# {{{
test = pd.read_csv(test_path)[['UserID', 'MovieID']].values
user_test  = test[:,0] - 1
movie_test = test[:,1] - 1
# }}}
# Load Model #
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
model_list = [
    '../model/35.h5',
    # '../model/36.h5',
    # '../model/40.h5',
    '../model/42.h5',
    '../model/43.h5',
    '../model/44.h5',
    '../model/45.h5',
    '../model/46.h5',
]
pred_train = np.zeros((len(model_list), train.shape[0], 1))
# predict trained models# {{{
print('')
for i, m in enumerate(model_list):
    print('predicting model from {}'.format(m))
    # model = load_model(m, custom_objects={'RMSE': RMSE})
    # pred_train[i] = model.predict([train[:,0], train[:,1]], batch_size=LOAD_BATCH_SIZE)
    # np.save('{}.npy'.format(i), pred_train[i])
    pred_train[i] = np.load('{}.npy'.format(i))
print('')
pred_train = pred_train.reshape(pred_train.shape[1], pred_train.shape[0])
# }}}
# Train Train Model #
def generate_model():# {{{
    model = Sequential()
    model.add(Dense(1, kernel_initializer=Constant(0.33), bias_initializer=Constant(0.07), input_shape=(6,)))
    # model.add(Dense(6, input_shape=(6,)))
    # model.add(BatchNormalization(input_shape=(6,)))
    # model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
    model.summary()
    return model
model_model = generate_model()
# }}}
# model_model.fit(pred_train, train[:,2], epochs=100, batch_size=TRAIN_BATCH_SIZE, validation_split=0.1)
model_model.fit(pred_train, train[:,2], epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE)

# Predict Test Model #
pred_test = np.zeros((len(model_list), test.shape[0], 1))
# predict trained models# {{{
print('')
for i, m in enumerate(model_list):
    print('predicting model from {}'.format(m))
    # model = load_model(m, custom_objects={'RMSE': RMSE})
    # pred_test[i] = model.predict([test[:,0], test[:,1]])
    # np.save('test_{}.npy'.format(i), pred_test[i])
    pred_test[i] = np.load('test_{}.npy'.format(i))
print('')
pred_test = pred_test.reshape(pred_test.shape[1], pred_test.shape[0])
# }}}

y_pred = model_model.predict(pred_test)

# save to h5 & csv# {{{
print('Saving model to: {}'.format(model_path))
model.save(model_path)
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, pred_rate[0]), file=f)
f.close()
# }}}
