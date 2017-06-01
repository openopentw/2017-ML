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
# keras# {{{
import keras.backend as K
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Flatten, Embedding, Dropout, Dense
from keras.layers.merge import Add, Dot, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}
# Parameter #
ID = 8
print('ID = {}'.format(ID))
# SPLIT_NUM = 80000
EMBD_DIM = 100
# EPOCHS = 50
PATIENCE = 20
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
user_train  = train[:,0] - 1
movie_train = train[:,1] - 1
rate_train  = train[:,2]

user_size   = user_train.max() + 1
movie_size  = movie_train.max() + 1
# }}}
# shuffle train# {{{
np.random.seed(42)
indices = np.arange(user_train.size)
np.random.shuffle(indices)
user_train  = user_train[indices]
movie_train = movie_train[indices]
rate_train  = rate_train[indices]
# }}}
# split vali# {{{
# user_vali  = user_train[-SPLIT_NUM:]
# movie_vali = movie_train[-SPLIT_NUM:]
# rate_vali  = rate_train[-SPLIT_NUM:]

# user_train  = user_train[:-SPLIT_NUM]
# movie_train = movie_train[:-SPLIT_NUM]
# rate_train  = rate_train[:-SPLIT_NUM]
# }}}
# load test data# {{{
test = pd.read_csv(test_path)[['UserID', 'MovieID']].values
user_test  = test[:,0] - 1
movie_test = test[:,1] - 1
# }}}

# Keras #
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
def generate_model():# {{{
    user_input = Input(shape=[1])
    user_vec = Embedding(user_size, EMBD_DIM, embeddings_initializer='random_normal')(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(0.7)(user_vec)

    movie_input = Input(shape=[1])
    movie_vec = Embedding(movie_size, EMBD_DIM, embeddings_initializer='random_normal')(movie_input)
    movie_vec = Flatten()(movie_vec)
    movie_vec = Dropout(0.7)(movie_vec)

    merge_vec = Concatenate()([user_vec, movie_vec])
    hidden = Dense(150, activation='elu')(merge_vec)
    hidden = Dense(100, activation='elu')(hidden)
    hidden = Dense(100, activation='elu')(hidden)
    hidden = Dense(50, activation='elu')(hidden)
    output = Dense(1)(hidden)

    model = Model([user_input, movie_input], output)
    model.summary()
    return model
model = generate_model()
# }}}
# fit & predict# {{{
earlystopping = EarlyStopping(monitor='val_RMSE', patience=PATIENCE, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_RMSE', mode='min')

EPOCHS = 300
model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
model.fit([user_train, movie_train], rate_train, epochs=EPOCHS, batch_size=1024, validation_split=0.1, callbacks=[earlystopping, checkpoint])
# }}}
# load & predict & save# {{{
model.load_weights(weights_path)
y_pred = model.predict([user_test, movie_test])
print('Saving model to: {}'.format(model_path))
model.save(model_path)
# }}}
# save to csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, pred_rate[0]), file=f)
f.close()
# }}}
