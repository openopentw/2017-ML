#! python3
"""
@author: b04902053
"""

# import# {{{
import sys
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Embedding, LSTM, GRU
from keras.models import Sequential, load_model
from keras.models import Model
from keras.optimizers import Adam
# }}}
import f1score

# Argvs# {{{
# TRAIN_FILE = sys.argv[1]
# TEST_FILE = sys.argv[2]
# OUTPUT = sys.argv[3]

TRAIN_FILE = './data/new_train_data.csv'
TEST_FILE = './data/test_data.csv'
OUTPUT = './submission.csv'

EMBEDDING_DIM = 100
GLOVE = 'd:/ML_data/HW5/data/glove.6B.100d.txt'
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

# Load Train data# {{{
print('Loading Training data')

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

# Load Test Dataset# {{{
print('Loading Testing Data')

test_txts = []

f = open(TEST_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split(',')[1:]
    for rest in lines[i][1:]:
        lines[i][0] += rest
    test_txts += [ lines[i][0] ]
# }}}

# Tokenize# {{{
tokenizer = Tokenizer()
tokenizer.fit_on_texts(txts + test_txts)

word_index = tokenizer.word_index
# }}}

sentence_length = 100
padding = 20

# x_train_data# {{{
sequences = tokenizer.texts_to_sequences(txts)
x_train_data = pad_sequences(sequences)
x_train_data = x_train_data[:,::-1]
# }}}

# x_test
max_article_length = x_train_data.shape[1]
test_sequences = tokenizer.texts_to_sequences(test_txts)
x_test = pad_sequences(test_sequences, maxlen=max_article_length)

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
# select_vali = -500
select_vali = -1
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
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1score.f1score, f1score.thres_f1score])
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1score.f1score, f1score.thres_f1score])
    return model
model = generate_model()# }}}

# model.fit(x_train, y_train, batch_size=128, epochs=150, validation_data=(x_vali, y_vali))
model.fit(x_train, y_train, batch_size=128, epochs=150)

model.save('9.h5')

exec(open('./load_fit_vali.py').read())
# exec(open('./load_fit.py').read())

THRES = 0.33

y_prob = model.predict(x_test)

# save y_prob to y_prob.csv# {{{
np.savetxt('y_prob_9_.csv', y_prob)
# }}}

# threshold# {{{
y_prob_max = np.max(y_prob, axis=1).reshape(y_prob.shape[0], 1)
y_thres = y_prob_max * THRES

y_prob[y_prob > y_thres] = 1
y_prob[y_prob != 1] = 0
# }}}

# change to tags# {{{
y_tags = []
for i in range(y_prob.shape[0]):
    tag = []
    for j in range(38):
        if y_prob[i][j] == 1:
            tag += [int2tag[j]]
    y_tags += [tag]
# }}}

# save to subm.csv# {{{
f = open('submission9.csv', 'w')
print('"id","tags"', file=f)
for i, ts in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(ts):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
# }}}
