use_CPU = 0
# Use CPU# {{{
import os
if use_CPU == 1:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# }}}
# import# {{{
import numpy as np
import string
import sys
import json
from nltk.corpus import stopwords

# import keras# {{{
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
# }}}
# }}}

# parameters #
ID = 25
nb_epoch = 150
patience = 15

split_ratio = 0.1
embedding_dim = 100
batch_size = 128

# argv# {{{
# # sys# {{{
# train_path  = sys.argv[1]
# test_path   = sys.argv[2]
# output_path = sys.argv[3]
# # }}}

train_path  = './data/new_train_data.csv'
test_path   = './data/test_data.csv'
output_path = './subm/submission_{}.csv'.format(ID)

tag_path    = './data/tag_list'
json_path   = './data/word_index.json'

model_path  = './model/best_{}.h5'.format(ID)
# }}}

# Util #
def read_tag_list(path):# {{{
    f = open(path)
    lines = f.readlines()
    f.close()
    tag_lists = [s.rstrip() for s in lines]
    return tag_lists
# }}}
def read_data(path,training):# {{{
    print ('Reading data from ',path)
    with open(path,'r', encoding='utf8') as f:

        tags = []
        articles = []

        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            articles.append(article)

        if training :
            assert len(tags) == len(articles)
    return (tags,articles)
# }}}
def get_embedding_dict(path):# {{{
    embedding_dict = {}
    with open(path,'r', encoding='utf8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict
# }}}
def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):# {{{
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix
# }}}
def to_multi_categorical(tags,tags_list): # {{{
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data
# }}}
def new_split_data(X_seq, Y, split_ratio):# {{{
    indices = np.arange(X_seq.shape[0])
    np.random.shuffle(indices)

    X_data = X_seq[indices]
    Y_data = [ Y[i] for i in indices ]
    # Y_data = Y[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)
# }}}
def shorter(sequences, Y_data = []): # let sequences shorter# {{{
    # # numbers# {{{
    #   0, 102
    # 102, 204
    # 204, 306
    #==========#
    #   0, 104
    # 100, 206
    # 202, 306
    #==========#
    #   0, 124
    #----------#
    #   0, 104
    # 100, 226
    #----------#
    #   0, 104
    # 100, 206
    # 202, 306
    # }}}
    sequences = sequences[:,::-1]
    add_sequences = np.zeros((sequences.shape[0] * 2, sequences.shape[1]), dtype=int)
    add_i = 0
    for i in range(sequences.shape[0]):
        if sequences[i][124] != 0:
            new_sequences = np.zeros((1, sequences.shape[1]), dtype=int)
            new_sequences[0,:-100] = sequences[i][100:]
            sequences[i][104:] = 0
            if new_sequences[0,126] != 0:
                add_sequences[add_i][:106] = new_sequences[0,:106]
                if Y_data:
                    Y_data += [Y_data[i]]
                add_i += 1
                add_sequences[add_i][:-102] = new_sequences[0,102:]
                if Y_data:
                    Y_data += [Y_data[i]]
                add_i += 1
            else:
                add_sequences[add_i] = new_sequences
                if Y_data:
                    Y_data += [Y_data[i]]
                add_i += 1
    add_sequences = add_sequences[:add_i]

    sequences = np.append(sequences, add_sequences, axis=0)
    sequences = sequences[:,:127]
    sequences = sequences[:,::-1]
    return sequences, Y_data
# }}}

# Custom Metrices #
def f1_score(y_true,y_pred):# {{{
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
# }}}

# Main #
# preprocessing data #
# read training & testing data# {{{
tag_list = read_tag_list(tag_path)
(Y_data,X_data) = read_data(train_path,True)
(_, X_test) = read_data(test_path,False)
all_corpus = X_data + X_test
print ('Find %d articles.' %(len(all_corpus)))
# }}}
# load tokenizer data# {{{
# # generate tokenizer for all data# {{{
# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(all_corpus)
# word_index = tokenizer.word_index
# # }}}

tokenizer = Tokenizer()
with open(json_path) as data_file:
    word_index = json.load(data_file)
tokenizer.word_index = word_index
# }}}
# convert word sequences to index sequence# {{{
print ('Convert to index sequences.')
train_sequences = tokenizer.texts_to_sequences(X_data)
test_sequences = tokenizer.texts_to_sequences(X_test)
# }}}
# padding to equal length# {{{
print ('Padding sequences.')
train_sequences = pad_sequences(train_sequences)
max_article_length = train_sequences.shape[1]
test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
# }}}

# TODO: add this !!!
(X_train,Y_train),(X_val,Y_val) = new_split_data(train_sequences, Y_data, split_ratio)

# TODO: add this !!!
# shorter train & val & test# {{{
X_train, Y_train = shorter(X_train, Y_train)
X_val, Y_val = shorter(X_val, Y_val)
X_test, Y_test_trash = shorter(test_sequences)
# }}}
# TODO: add this !!!
# let stop_words become 0 in train & test# {{{
stop_sequences  = tokenizer.texts_to_sequences(stopwords.words('english'))
stop_sequences  = np.array([ seq[0] for seq in stop_sequences if seq ])
for s in stop_sequences:
    X_train[X_train == s] = 0
    X_val[X_val == s] = 0
    X_test[X_test == s]  = 0
# }}}
# TODO: add this !!!
# to_categorical# {{{
Y_train = to_multi_categorical(Y_train,tag_list)
Y_val = to_multi_categorical(Y_val,tag_list)
# }}}
# get mebedding matrix from glove# {{{
print ('Get embedding dict from glove.')
embedding_dict = get_embedding_dict('d:/ML_data/HW5/data/glove.6B.%dd.txt'%embedding_dim)
print ('Found %s word vectors.' % len(embedding_dict))
num_words = len(word_index) + 1
print ('Create embedding matrix.')
embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)
# }}}

# training model #
# generate model# {{{
print ('Generating model.')
model = Sequential()
model.add(Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=127, trainable=False))
model.add(GRU(128,activation='tanh',dropout=0.5))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(38,activation='sigmoid'))
model.summary()

adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1_score])
# }}}
# fit# {{{
earlystopping = EarlyStopping(monitor='val_f1_score', patience=patience, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath='best_weights{}.h5'.format(ID), verbose=1, save_best_only=True, save_weights_only=True, monitor='val_f1_score', mode='max')
hist = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=nb_epoch, batch_size=batch_size, callbacks=[earlystopping,checkpoint])
# }}}
# load best# {{{
print('Loading best model...')
model.load_weights('./best_weights{}.h5'.format(ID))
Y_pred = model.predict(X_test)
thresh = 0.4
# }}}
# predict# {{{
print('Saving submission to {}'.format(output_path))
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
# }}}

print('Saving model to {}'.format(model_path))
model.save(model_path)
