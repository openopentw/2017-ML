"""
@ author: b04902053
"""

use_device = 'GPU'  # CPU / GPU
# Use CPU# {{{
import os
if use_device == 'CPU':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
elif use_device == 'GPU':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
ID = 2
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

user_size = train['UserID'].unique().size + 1
movie_size = train['MovieID'].unique().max() + 1

train = train[['UserID', 'MovieID', 'Rating']].values
user_id = train[:,0]
# user_id = user_id.reshape(user_id.size, 1)
movie_id = train[:,1]
# movie_id = movie_id.reshape(movie_id.size, 1)
rate = train[:,2]
# rate = rate.reshape(rate.size, 1)
# }}}
# load test data# {{{
test = pd.read_csv(test_path).values[:,1:]
test_user = test[:,0]
test_movie = test[:,1]
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
    user_vec = Embedding(user_size, 100)(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = Dropout(0.5)(user_vec)

    movie_input = Input(shape=[1])
    movie_vec = Embedding(movie_size, 100)(movie_input)
    movie_vec = Flatten()(movie_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    dot_vec = Dot(axes=1)([user_vec, movie_vec])

    model = Model([user_input, movie_input], dot_vec)

    model.summary()

    model.compile(loss='mse', optimizer='adam')
    return model
model = generate_model()
# }}}

# model.fit([user_id, movie_id], y_matrix, validation_data=(X_val, Y_val), epochs=20, batch_size=batch_size, callbacks=[earlystopping,checkpoint])
# model.fit([user_id, movie_id], y_matrix, epochs=20)
model.fit([user_id, movie_id], rate, epochs=200, batch_size=2048)
y_pred = model.predict([test_user, test_movie])

print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, np.round(pred_rate[0])), file=f)
f.close()
