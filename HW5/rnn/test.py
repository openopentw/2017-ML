# import# {{{
import numpy as np
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
ID = 14
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

train_path  = '../data/new_train_data.csv'
test_path   = '../data/test_data.csv'
output_path = './submission_{}.csv'.format(ID)

tag_path    = './tag_list'
json_path   = './word_index.json'

model_path  = './best_{}.h5'.format(ID)
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
# let stop_words become 0 in train & test# {{{
stop_sequences  = tokenizer.texts_to_sequences(stopwords.words('english'))
stop_sequences  = np.array([ seq[0] for seq in stop_sequences if seq ])
for s in stop_sequences:
    train_sequences[train_sequences == s] = 0
    test_sequences [test_sequences == s]  = 0
# }}}
# to_categorical & split validation set# {{{
train_tag = to_multi_categorical(Y_data,tag_list)
(X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)
# }}}

# training model #
# load best# {{{
# print('Loading best model...')
# model.load_weights('./best_weights{}.h5'.format(ID))
model = load_model(model_path, custom_objects={"f1_score": f1_score})
Y_pred = model.predict(test_sequences)
# }}}
# predict# {{{
thresh = 0.4
print('Saving submission to {}'.format(output_path))
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
# }}}

# print('Saving model to {}'.format(model_path))
# model.save(model_path)
