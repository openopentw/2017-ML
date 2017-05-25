# parameters #
ID = 301
# argv# {{{
# # sys# {{{
# train_path  = sys.argv[1]
# test_path   = sys.argv[2]
# output_path = sys.argv[3]
# # }}}
train_path   = './data/new_train_data.csv'
test_path    = './data/test_data.csv'
output_path  = './subm/submission_{}.csv'.format(ID)

# tag_path     = './data/tag_list'
pickle_path  = './data/uu_tokenizer.pickle'
# }}}


import re
from nltk.corpus import stopwords
import pickle
import sys,time
import numpy as np
# data 40220 unique after preprocess

def load_data(seperation=True):# {{{
    label_sets=[]
    text_sets=[]
    dic = {}
    dic1 = {}
    test_sets=[]
    for string in open(test_path,'r', encoding='utf8').readlines()[1:]:
        num,words=string.split(',',1)
        words = " ".join([word for word in words.split() if "http" not in word])
        # Remove all garbage punctuation and turn lower split
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        # Remove stop words
        words = [w for w in words if w not in stopwords.words("english")]
        for i in words:
            dic[i] = True
        # Join back to string ?
        #test_sets.append( words if seperation else " ".join(words))            
        test_sets.append( words )            
    
    for string in open(train_path,'r', encoding='utf8').readlines()[1:]:
        num,label,words= string.split(',',2)
        words = " ".join([word for word in words.split() if "http" not in word])
        # Preprocess for label
        label_sets.append(label[1:-1].split())

        # Remove all garbage punctuation and turn lower split
        words = re.sub("[^a-zA-Z]"," ",words).lower().split()
        # Remove stop words
        for i in words:
            dic1[i] = True
        words = [w for w in words if w not in stopwords.words("english") and w in dic]
        # Join back to string ?
        text_sets.append( words if seperation else " ".join(words))            
    new_test_sets = []
    for string in test_sets:
        words = [word for word in string if word in dic1]
        new_test_sets.append( words if seperation else " ".join(words))            
    return np.array(text_sets),np.array(label_sets),new_test_sets
# }}}

print('finish loading data')

# """
# Save as pickle object
"""
with open(pickle_path,'wb') as f:
    pickle.dump(load_data(seperation=False),f)
"""
f = open(pickle_path,'rb')
texts,labels,test_data = pickle.load(f)
f.close()

"""
f2 = open('data_sep','rb', encoding='utf8')
t2, labels = pickle.load(f2)
"""
# =======================================

indices = np.arange(texts.shape[0])
np.random.shuffle(indices)
texts = texts[indices]
labels = labels[indices]

# =======================================

print('start training')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC,SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
mul = MultiLabelBinarizer()
y_enc = mul.fit_transform(labels)
par =mul.get_params()
classifier = Pipeline([
    ('vectorizer',CountVectorizer(analyzer ="word", tokenizer = None, preprocessor = None, stop_words = None, max_features =30000)),
    ('tfidf',TfidfTransformer()),
    ('clf',OneVsRestClassifier(XGBClassifier(seed = 7122,scale_pos_weight=0.5)))])
train_x,test_x,train_y,test_y = train_test_split(texts,y_enc,test_size=0.2)
classifier.fit(texts,y_enc)
#classifier.fit(train_x,train_y)
predicted = classifier.predict(test_data)
#predicted = classifier.predict(test_x)

print('saving subm.csv to {}'.format(output_path))

#my_metrics = metrics.classification_report(test_y,predicted)
#print(my_metrics)
with open(output_path,'w') as fd:
    print("id,tags",file=fd)
    for index,text in enumerate(mul.inverse_transform(predicted)):
        print(index,",\""," ".join(text),"\"",sep='',file=fd)
# C= 1e-2 Eout = 49 random_state = 7122
