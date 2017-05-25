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

# parameter #
ID = 14

# argv# {{{
# # sys# {{{
# train_path  = sys.argv[1]
# test_path   = sys.argv[2]
# output_path = sys.argv[3]
# # }}}

train_path  = '../data/new_train_data.csv'
test_path   = '../data/test_data.csv'
output_path = '../subm/submission_{}_load.csv'.format(ID)

tag_path    = '../data/tag_list'
json_path   = '../data/word_index.json'

model_path  = '../model/best_{}.h5'.format(ID)
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
# read training and testing data# {{{
tag_list = read_tag_list(tag_path)
(Y_data,X_data) = read_data(train_path,True)
(_, X_test) = read_data(test_path,False)
all_corpus = X_data + X_test
print ('Find %d articles.' %(len(all_corpus)))
# }}}
# tokenizer for all data# {{{
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

# load model# {{{
print('Loading best model: {}'.format(model_path))
model = load_model(model_path, custom_objects={"f1_score": f1_score})
Y_pred = model.predict(test_sequences)
thresh = 0.4
# }}}

# predict#{{{
print('Predict & save to {}'.format(output_path))
with open(output_path,'w') as output:
    print ('\"id\",\"tags\"',file=output)
    Y_pred_thresh = (Y_pred > thresh).astype('int')
    for index,labels in enumerate(Y_pred_thresh):
        labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
        labels_original = ' '.join(labels)
        print ('\"%d\",\"%s\"'%(index,labels_original),file=output)
# }}}
