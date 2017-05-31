#! python3
"""
@ author: b04902053
"""

# import# {{{
import pandas as pd
import numpy as np
# keras
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.layers.merge import Add, Dot, Concatenate
# }}}
# Parameter #
ID = 1
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
userid = train['UserID'].unique()
movieid = train['MovieID'].unique()
matrix = np.zeros((userid.max(), movieid.max()), dtype=int)
train = train[['UserID', 'MovieID', 'Rating']].values
for data in train:
    matrix[data[0]-1, data[1]-1] = data[2]
# }}}
'''
# load test data# {{{
x_test_data = pd.read_csv(test_path).values[:,1:]
# }}}
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
'''
def generate_model():

    user_embd = Embedding(user.shape[0], 100, trainable=True)

    return model
'''
