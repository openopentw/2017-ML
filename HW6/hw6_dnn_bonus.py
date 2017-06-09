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
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Dot, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}
# Parameter #
ID = 51
print('ID = {}'.format(ID))
EMBD_DIM = 100
BATCH_SIZE = 8192
PATIENCE = 100
EPOCHS = 1000
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
train[:,0]  = train[:,0] - 1
train[:,1] = train[:,1] - 1

user_size   = train[:,0].max() + 1
movie_size  = train[:,1].max() + 1
# }}}
# shuffle train# {{{
np.random.seed(42)
indices = np.arange(train.shape[0])
np.random.shuffle(indices)
train = train[indices]
# user_train  = user_train[indices]
# movie_train = movie_train[indices]
# rate_train  = rate_train[indices]
# }}}
# load test data# {{{
test = pd.read_csv(test_path)[['UserID', 'MovieID']].values
user_test  = test[:,0] - 1
movie_test = test[:,1] - 1
# }}}
# load user age# {{{
user_age = pd.read_csv(user_path, delimiter='::')[['UserID', 'Age']]
user_age = user_age.sort(['UserID', 'Age']).values
user_age[:,0] -= 1
user_age = user_age[:,1]
emb_age = np.zeros(train[:,0].size, dtype=int)
for i in range(emb_age.size):
    emb_age[i] = user_age[ train[i,0] ]
test_emb_age = np.zeros(user_test.size, dtype=int)
for i in range(test_emb_age.size):
    test_emb_age[i] = user_age[ int(user_test[i]) ]
# }}}
# Keras #
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
def generate_model():# {{{
    user_input = Input(shape=[1])
    user_vec = Embedding(user_size, EMBD_DIM)(user_input)
    user_vec = Flatten()(user_vec)
    # user_vec = BatchNormalization()(user_vec)
    user_vec = Dropout(0.5)(user_vec)

    user_age_input = Input(shape=[1])

    movie_input = Input(shape=[1])
    movie_vec = Embedding(movie_size, EMBD_DIM)(movie_input)
    movie_vec = Flatten()(movie_vec)
    # movie_vec = BatchNormalization()(movie_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    merge_vec = Concatenate()([user_vec, user_age_input, movie_vec])
    hidden = Dense(2048, activation='relu')(merge_vec)
    hidden = Dropout(0.5)(hidden)
    output = Dense(1)(hidden)

    model = Model([user_input, user_age_input, movie_input], output)
    model.summary()
    return model
model = generate_model()
# }}}
# fit & predict# {{{
earlystopping = EarlyStopping(monitor='val_RMSE', patience=PATIENCE, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_RMSE', mode='min')

model.compile(loss='mse', optimizer='adam', metrics=[RMSE])
model.fit([train[:,0], emb_age, train[:,1]], train[:,2], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1, callbacks=[earlystopping, checkpoint])
# model.fit([train[:,0], train[:,1]], train[:,2], epochs=EPOCHS, batch_size=16000)
# }}}
# load & predict & save# {{{
# model.load_weights(weights_path)
y_pred = model.predict([user_test, test_emb_age, movie_test])
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

'''
score = model.evaluate([train[:,0], train[:,1]], train[:,2], batch_size=1024)
f = open('dnn_score.txt', 'w')
print(score, file=f)
f.close()
'''
