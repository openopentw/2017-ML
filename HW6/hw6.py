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
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}
# Parameter #
# ID = 40
ID = int(sys.argv[1])
print('ID = {}'.format(ID))

# EPOCHS = 1500
EPOCHS = int(sys.argv[2])
EMBD_DIM = 150
SEED = 87

NORM = False
USER_NORM = False   # NORM or USER_NORM, only one can be True
BIAS = False

VALI = False
if VALI == True:
    PATIENCE = 200
# argv# {{{
train_path  = './data/train.csv'
test_path   = './data/test.csv'
output_path = './subm/submission_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))

user_path   = './data/users.csv'
movie_path  = './data/movies.csv'

weights_path = './weights/weights_{}.h5'.format(ID)
model_path   = './model/{}.h5'.format(ID)
print('Will save model to: {}'.format(model_path))
# }}}
# Before Train #
# load train data# {{{
train = pd.read_csv(train_path)[['UserID', 'MovieID', 'Rating']].values
train[:,0] = train[:,0] - 1
train[:,1] = train[:,1] - 1
user_size  = train[:,0].max() + 1
movie_size = train[:,1].max() + 1
# }}}
# shuffle train# {{{
np.random.seed(SEED)
indices = np.arange(train.shape[0])
np.random.shuffle(indices)
train = train[indices]
# }}}
# load test data# {{{
test = pd.read_csv(test_path)[['UserID', 'MovieID']].values
user_test  = test[:,0] - 1
movie_test = test[:,1] - 1
# }}}
# the 2 methods of normalizing on rating# {{{
train = train.astype(float)
if NORM == True:
    mean = np.mean(train[:,2])
    std  = np.std(train[:,2])
    train[:,2] = (train[:,2] - mean) / std
elif USER_NORM == True:
    mean = np.zeros(user_size)
    std  = np.zeros(user_size)
    for i in range(user_size):
        mean[i] = np.mean(train[ train[:,0]==i , 2])
        std[i]  = np.std (train[ train[:,0]==i , 2]) + 1e-8
        train[train[:,0] == i, 2] = (train[train[:,0] == i, 2] - mean[i]) / std[i]
        if i % 100 == 0:
            print(i, end=',')
# }}}

# Keras #
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
def generate_model():# {{{
    user_input = Input(shape=[1])
    user_vec = Embedding(user_size, EMBD_DIM)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = BatchNormalization()(user_vec)
    user_vec = Dropout(0.4)(user_vec)

    movie_input = Input(shape=[1])
    movie_vec = Embedding(movie_size, EMBD_DIM)(movie_input)
    movie_vec = Flatten()(movie_vec)
    movie_vec = BatchNormalization()(movie_vec)
    movie_vec = Dropout(0.4)(movie_vec)

    dot_vec = Dot(axes=1)([user_vec, movie_vec])

    if BIAS == True:
        user_bias = Embedding(user_size, 1, embeddings_initializer='zeros')(user_input)
        user_bias = Flatten()(user_bias)
        movie_bias = Embedding(movie_size, 1, embeddings_initializer='zeros')(movie_input)
        movie_bias = Flatten()(movie_bias)
        dot_vec = Add()([dot_vec, user_bias, movie_bias])

    model = Model([user_input, movie_input], dot_vec)
    model.summary()

    return model
model = generate_model()
# }}}

# fit & load# {{{
if VALI == True:
    earlystopping = EarlyStopping(monitor='val_RMSE', patience=PATIENCE, verbose=1, mode='min')
    checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_RMSE', mode='min')

model.compile(loss='mse', optimizer='adam', metrics=[RMSE])

if VALI == True:
    model.fit([train[:,0], train[:,1]], train[:,2], epochs=EPOCHS, batch_size=10000, validation_split=0.1, callbacks=[checkpoint, earlystopping])
    model.load_weights(weights_path)
else:
    model.fit([train[:,0], train[:,1]], train[:,2], epochs=EPOCHS, batch_size=10000)
# }}}
# predict & normalize# {{{
y_pred = model.predict([user_test, movie_test])
if NORM == True:
    y_pred = y_pred * std + mean
elif USER_NORM == True:
    for i in range(user_size):
        y_pred[user_test == i, 0] = y_pred[user_test == i, 0] * std[i] + mean[i]
        if i % 100 == 0:
            print(i, end=',')
# }}}
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

# for run many times use# {{{
score = model.evaluate([train[:,0], train[:,1]], train[:,2], batch_size=1024)
f = open('score.txt', 'a')
print(ID, end='\t', file=f)
print(score, file=f)
f.close()
# }}}
