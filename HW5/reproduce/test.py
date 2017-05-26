# import# {{{
import numpy as np
import string
import sys
# keras# {{{
import keras.backend as K 
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,BatchNormalization# }}}
import pickle
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import re
import random
from numpy import genfromtxt# }}}
ID = 18
# argv# {{{
# train_path = sys.argv[1]
# test_path = sys.argv[2]
train_path = '../data/train_data.csv'
test_path = '../data/test_data.csv'
output_path = './submission_{}.csv'.format(ID)

csv_path = '../vote/submission_vote_{}.csv'.format(ID)
model_path = './model_{}.h5'.format(ID)
weights_path = './weights_{}.h5'.format(ID)

tag_path = './tag_list'
tokenizer_path = './tokenizer.pickle'
# }}}
###   parameter   ###
EPOCHS = 30
embedding_dim = 100
batch_size = 200
VALIDATION_SPLIT = 0.5
patience = 5
#   Util   #
def read_data(path,training):# {{{
    print ('Reading data from ',path)
    stopword = stopwords.words('english')
    lmtzr = WordNetLemmatizer()
    with open(path,'r', encoding='utf8') as f:

        tags = []
        articles = []
        tags_list = []

        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]

                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)

                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]

            article = re.sub('[^a-zA-Z]', ' ', article)
            article = text_to_word_sequence(article, lower=True, split=' ')
            article = [ w for w in article if w not in stopword ]
            article = [ lmtzr.lemmatize(w) for w in article ]
            article = ' '.join(article)
            articles.append(article)

        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)
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
def split_data(X,Y,split_ratio):# {{{
    indices = np.arange(X.shape[0]) 
    random.seed(42) 
    np.random.shuffle(indices) 

    X_data = X[indices]
    Y_data = Y[indices]

    num_validation_sample = int(split_ratio * X_data.shape[0] )

    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)
# }}}
def f1_score(y_true,y_pred):# {{{
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)

    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))
# }}}
#   Main function   #
# read tags_list# {{{
f = open(tag_path, encoding='utf8')
lines = f.readlines()
f.close()
tags_list = [s.rstrip() for s in lines]
print (tags_list)
# }}}
(_,X_data,_) = read_data(test_path,False)

tokenizer = pickle.load(open(tokenizer_path,'rb'))
print ('Convert to index sequences.')
X_test = tokenizer.texts_to_matrix(X_data, mode='tfidf')

print ('Building model.')

model = Sequential()
model.add(Dense(512,activation='elu',input_dim=40587))
model.add(Dense(512,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(512,activation='elu'))
model.add(Dense(38,activation='sigmoid'))
model.summary()
adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=[f1_score])

model.load_weights(weights_path)
print('loading weights from {}'.format(model_path))

# predict #
# load best# {{{
# print('Loading best weights from {}'.format(weights_path))
# model.load_weights(weights_path)
Y_pred = model.predict(X_test)
thresh = 0.33
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
