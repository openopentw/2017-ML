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
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Flatten, Embedding, Dropout, merge
from keras.layers.merge import Add, Dot, Concatenate
# }}}
# }}}
# Parameter #
ID = 4
split_num = 80000
# argv# {{{
train_path  = './data/train.csv'
test_path   = './data/test.csv'
output_path = './subm/submission_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))

user_path   = './data/users.csv'
movie_path  = './data/movies.csv'
# }}}
# Before Train #
# load train data# {{{
train = pd.read_csv(train_path)

user_size  = train['UserID'].unique().size
movie_size = train['MovieID'].unique().max()

train = train[['UserID', 'MovieID', 'Rating']].values
user_train  = train[:,0]
movie_train = train[:,1]
rate_train  = train[:,2]
# user_train  = user_train.reshape(user_train.size, 1)
# movie_train = movie_train.reshape(movie_train.size, 1)
# rate_train  = rate_train.reshape(rate.size, 1)
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
# user_vali  = user_train[-split_num:]
# movie_vali = movie_train[-split_num:]
# rate_vali  = rate_train[-split_num:]

# user_train  = user_train[:-split_num]
# movie_train = movie_train[:-split_num]
# rate_train  = rate_train[:-split_num]
# }}}
# load test data# {{{
test = pd.read_csv(test_path).values[:,1:]
user_test  = test[:,0]
movie_test = test[:,1]
# }}}
'''
# load user data# {{{
user_data = pd.read_csv(user_path).values
user_data[user_data == 'F'] = 0
user_data[user_data == 'M'] = 1
# TODO: notice zip-code
# TODO: combine with train & test data
# }}}
# load movie data# {{{
# movie_data = pd.read_csv(movie_path)
f = open(movie_path, encoding='latin1')
movie_data = f.readlines()[1:]
f.close()

movie = []
genres
for i in range(len(movie_data)):
    movie_data[i] = movie_data[i][:-1]

    start = movie_data[i].find(',') + 1
    delimiter = movie_data[i].find('),') + 1

    title = movie_data[i][start:delimiter]
    genre = movie_data[i][delimiter+1:]].split('|')
    movie += [title, genre]
    genres += [[genre]]

    # TODO: if no bug, remove it
    if len(movie_data[i]) != 2:
        print('BUG length: i={}, len={}'.format(i, len(movie_data[i])))
        break
# }}}
'''

# Keras #
def generate_model():# {{{
    user_input = Input(shape=[1])
    user_vec = Embedding(user_size + 1, 100)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(0.4)(user_vec)

    movie_input = Input(shape=[1])
    movie_vec = Embedding(movie_size + 1, 100)(movie_input)
    movie_vec = Flatten()(movie_vec)
    movie_vec = Dropout(0.4)(movie_vec)

    dot_vec = Dot(axes=1)([user_vec, movie_vec])

    model = Model([user_input, movie_input], dot_vec)

    model.summary()

    model.compile(loss='mse', optimizer='adam')
    return model
model = generate_model()
# }}}

# model.fit([user_id, movie_id], y_matrix, validation_data=(X_val, Y_val), epochs=20, batch_size=batch_size, callbacks=[earlystopping,checkpoint])
# earlystopping = EarlyStopping(monitor='val_f1_score', patience=patience, verbose=1, mode='max')
# checkpoint = ModelCheckpoint(filepath='best_weights{}.h5'.format(ID), verbose=1, save_best_only=True, save_weights_only=True, monitor='val_f1_score', mode='max')
# model.fit([user_train, movie_train], rate_train, epochs=50, batch_size=1024, validation_data=([user_vali, movie_vali], rate_vali))
model.fit([user_train, movie_train], rate_train, epochs=50, batch_size=1024)
y_pred = model.predict([user_test, movie_test])

# save to csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, np.round(pred_rate[0])), file=f)
f.close()
# }}}
