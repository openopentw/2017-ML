#! python3
"""
@author: b04902053
"""

ID = 28
patience = 150

# parameters #
output_path  = './subm/submission_{}.csv'       .format(ID)
model_path   = './model/best_{}.h5'             .format(ID)
weights_path = './best_weights_{}.h5'           .format(ID)
json_path    = './word_index/word_index_{}.json'.format(ID)

# import# {{{
import sys
import numpy as np
import json

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding, LSTM, GRU
from keras.models import Sequential, load_model
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# import f1score
from old_py import f1score

# Argvs# {{{
# TRAIN_FILE = sys.argv[1]
# TEST_FILE = sys.argv[2]
# OUTPUT = sys.argv[3]

TRAIN_FILE = './data/new_train_data.csv'
TEST_FILE = './data/test_data.csv'
OUTPUT = './submission.csv'

EMBEDDING_DIM = 100
GLOVE = 'd:/ML_data/HW5/data/glove.6B.100d.txt'

tag_path  = './data/tag_list'
# }}}

# Load Glove# {{{
print('Indexing word vectors.')

glove = {}
f = open(GLOVE, 'r', encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    glove[word] = coefs
f.close()

print('Found %s word vectors.' % len(glove))
# }}}

# Load dataset# {{{
print('Processing text dataset')

all_tags = []
tags = []
txts = []

f = open(TRAIN_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split('"')[1:]
    for rest in lines[i][2:]:
        lines[i][1] += rest

    line_tag = lines[i][0].split(' ')
    all_tags += line_tag
    tags += [line_tag]
    txts += [ lines[i][1] ]
# }}}

# load tokenizer data# {{{

# generate tokenizer for all data# {{{
tokenizer = Tokenizer()
tokenizer.fit_on_texts(txts)
sequences = tokenizer.texts_to_sequences(txts)
with open(json_path, 'w') as fp:
    json.dump(tokenizer.word_index, fp)
# }}}
# tokenizer = Tokenizer()
# with open(json_path) as data_file:
    # word_index = json.load(data_file)
# tokenizer.word_index = word_index

sequences = tokenizer.texts_to_sequences(txts)
# }}}

# x_train_data# {{{
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

x_train_data = pad_sequences(sequences, maxlen=313)
# }}}

def read_tag_list(path):# {{{
    f = open(path)
    lines = f.readlines()
    f.close()
    tag_lists = [s.rstrip() for s in lines]
    return tag_lists
all_tags = read_tag_list(tag_path)
# }}}

# y_train_data# {{{
all_tags = set(all_tags)
tag2int = {tag:i for i, tag in enumerate(all_tags)}
int2tag = {i:tag for i, tag in enumerate(all_tags)}

y_train_data = np.zeros((len(tags), len(all_tags)))
for i in range(len(tags)):
    int_tags = [tag2int[t] for t in tags[i]]
    y_train_data[i, int_tags] = 1
# }}}

#  Split train & vali# {{{
select_vali = -500
# select_vali = -1
x_train = x_train_data[:select_vali]
y_train = y_train_data[:select_vali]
x_vali  = x_train_data[select_vali:]
y_vali  = y_train_data[select_vali:]
# }}}

# Prepare Embedding Matrix# {{{
print('Preparing embedding matrix.')
num_words = len(word_index)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= len(word_index):
        continue
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
# }}}

def generate_model():   # with execute# {{{
    model = Sequential()

    model.add(Embedding(num_words, EMBEDDING_DIM, weights=[embedding_matrix], input_length=x_train.shape[1], trainable=False))

    # model.add(Conv1D(128, 5, activation='elu'))
    # model.add(AveragePooling1D(5))
    # model.add(Dropout(0.25))
    # model.add(Conv1D(128, 5, activation='elu'))
    # model.add(AveragePooling1D(5))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    model.add(GRU(128, activation='elu', dropout=0.3))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(all_tags), activation='sigmoid'))

    model.summary()

    adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1score.thres_f1score])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1score.f1score, f1score.thres_f1score])
    return model
model = generate_model()# }}}

# model.fit(x_train, y_train, batch_size=128, epochs=150, validation_data=(x_vali, y_vali))
# model.fit(x_train, y_train, batch_size=128, epochs=1, validation_data=(x_vali, y_vali))
# model.fit(x_train, y_train, batch_size=128, epochs=100)

# earlystopping = EarlyStopping(monitor='val_thres_f1score', patience=patience, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='best_weights_{}.h5'.format(ID), verbose=1, save_best_only=True, save_weights_only=True, monitor='val_thres_f1score', mode='max')
model.fit(x_train, y_train, batch_size=128, epochs=150, validation_data=(x_vali, y_vali), callbacks=[checkpoint])
# model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=nb_epoch, batch_size=batch_size, callbacks=[earlystopping,checkpoint])

# print('Loading best model weights from: {}'.format(weights_path))
# model.load_weights(weights_path)

exec(open('./old_py/load_fit_vali.py').read())
exec(open('./old_py/load_fit.py').read())

print('Saving model to: {}'.format(model_path))
model.save(model_path)
