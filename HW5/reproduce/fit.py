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
v1 =[]
f1 = open(csv_path,"r", encoding='utf8')
f1.readline()
for line in f1:
    cnt = 0
    pos = line.find('\"')
    pos = line.find('\"',pos+1)
    end = len(line) -2
    tag = line[pos+3:end].split(' ')
    v1.append(tag)

f = open(tag_path, encoding='utf8')
lines = f.readlines()
f.close()
tags_list = [s.rstrip() for s in lines]

# tags_list = ['SCIENCE-FICTION', 'SPECULATIVE-FICTION', 'FICTION', 'NOVEL', 'FANTASY', "CHILDREN'S-LITERATURE", 'HUMOUR', 'SATIRE', 'HISTORICAL-FICTION', 'HISTORY', 'MYSTERY', 'SUSPENSE', 'ADVENTURE-NOVEL', 'SPY-FICTION', 'AUTOBIOGRAPHY', 'HORROR', 'THRILLER', 'ROMANCE-NOVEL', 'COMEDY', 'NOVELLA', 'WAR-NOVEL', 'DYSTOPIA', 'COMIC-NOVEL', 'DETECTIVE-FICTION', 'HISTORICAL-NOVEL', 'BIOGRAPHY', 'MEMOIR', 'NON-FICTION', 'CRIME-FICTION', 'AUTOBIOGRAPHICAL-NOVEL', 'ALTERNATE-HISTORY', 'TECHNO-THRILLER', 'UTOPIAN-AND-DYSTOPIAN-FICTION', 'YOUNG-ADULT-LITERATURE', 'SHORT-STORY', 'GOTHIC-FICTION', 'APOCALYPTIC-AND-POST-APOCALYPTIC-FICTION', 'HIGH-FANTASY']
print (tags_list)
(_,X_data,_) = read_data(test_path,False)

ans = np.zeros((1234,38))


tokenizer = pickle.load(open(tokenizer_path,'rb'))
print ('Convert to index sequences.')
test_matrix = tokenizer.texts_to_matrix(X_data,mode='tfidf')

for x in range(1234):
    for y in v1[x]:
        for z in range(38):
            if y== tags_list[z]:
                ans[x][z]=1




(Y_data,X_data,_) = read_data(train_path,True)
train_matrix = tokenizer.texts_to_matrix(X_data, mode = 'tfidf')
train_tag = to_multi_categorical(Y_data,tags_list) 
(X_train,Y_train),(X_val,Y_val) = split_data(train_matrix,train_tag,VALIDATION_SPLIT)




test_matrix = np.append(test_matrix, X_train, axis = 0)

ans = np.append(ans, Y_train, axis = 0)
print(np.shape(ans))

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

earlystopping = EarlyStopping(monitor='f1_score', patience=patience, verbose=1, mode='max')
checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='f1_score', mode='max')
hist = model.fit(test_matrix, ans, epochs=30, validation_data=(X_val, Y_val), batch_size=batch_size, callbacks=[earlystopping, checkpoint])

# model.load_weights(weights_path)
print('model saved at {}'.format(model_path))
# model.save(model_path)
