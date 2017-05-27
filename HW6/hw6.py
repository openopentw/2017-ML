#! python3
"""
@ author: b04902053
"""

# import# {{{
import pandas as pd
import numpy as np
# }}}
# Parameter #
ID = 1
# argv# {{{
train_path  = './data/train.csv'
test_path   = './data/test.csv'
output_path = './subm/submission_{}.csv'.format(ID)

user_path   = './data/users.csv'
movie_path  = './data/movies.csv'
# }}}
# Before Train #
# load train data# {{{
train_data = pd.read_csv(train_path).values
x_train_data = train_data[:,:-1]
y_train_data = train_data[:,-1]
# }}}
# load test data# {{{
x_test_data = pd.read_csv(test_path).values
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
movie_data = pd.read_csv(movie_data)
# }}}
'''

# Train #
