use_CPU = True
# Use CPU# {{{
import os
if use_CPU == True:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# }}}
# import# {{{
import numpy as np
import sys
import pickle
import re
import xgboost as xgb
# nltk# {{{
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
# }}}
# keras# {{{
import keras.backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Flatten
from keras.preprocessing.text import text_to_word_sequence
# }}}
# }}}
# parameters #
ID = 101
# ID = sys.argv[1]
print('')
print('ID = {}'.format(ID))
print('')
nb_epoch = 60
patience = 5

split_ratio = 0.1
embedding_dim = 100
batch_size = 64
# argv# {{{
# # sys# {{{
# train_path  = sys.argv[1]
# test_path   = sys.argv[2]
# output_path = sys.argv[3]
# # }}}
train_path   = './data/new_train_data.csv'
test_path    = './data/test_data.csv'
output_path  = './subm/submission_{}.csv'.format(ID)

tag_path     = './data/tag_list'
pickle_path  = './data/tokenizer.pickle'

weights_path = './weights/weights_{}.h5'.format(ID)
# model_path   = './model/{}.h5'.format(ID)
model_path   = 'd:/ML_data/HW5/model/{}.h5'.format(ID)
# }}}

# Util #
def read_tag_list(path):# {{{
    f = open(path)
    lines = f.readlines()
    f.close()
    tag_lists = [s.rstrip() for s in lines]
    return tag_lists
# }}}
stopword = stopwords.words('english')
lmtzr = WordNetLemmatizer()
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

            # preprocessing texts
            article = re.sub('[^a-zA-Z]', ' ', article)
            article = text_to_word_sequence(article, lower=True, split=' ')
            article = [ w for w in article if w not in stopword ]
            article = [ lmtzr.lemmatize(w) for w in article ]
            article = ' '.join(article)

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
    np.random.seed(42)
    indices = np.arange(X.shape[0])
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

# Custom Metrices #
def f1_score(y_true,y_pred):# {{{
    thresh = 0.33
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
# }}}
# load tokenizer data# {{{
# # generate tokenizer for all data# {{{
# tokenizer = Tokenizer()
# all_corpus = X_data + X_test
# tokenizer.fit_on_texts(all_corpus)
# pickle.dump(tokenizer, open(pickle_path, 'wb'))
# # }}}
tokenizer = pickle.load(open(pickle_path, 'rb'))
# }}}
# convert word sequences to matrix# {{{
print ('Convert to bag matrix.')
X_train = tokenizer.texts_to_matrix(X_data, mode='tfidf')
X_test = tokenizer.texts_to_matrix(X_test, mode='tfidf')
# }}}
# to_categorical & split validation set# {{{
train_tag = to_multi_categorical(Y_data,tag_list)
(X_train,Y_train),(X_val,Y_val) = split_data(X_train,train_tag,split_ratio)
# }}}


dtrain = xgb.DMatrix(X_train, Y_train)
# dval = xgb.DMatrix(X_val, Y_val)
# dtest = xgb.DMatrix(X_test)

xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 1.0,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
