"""
@author: b04902053
"""
use_device = 'cpu'  # cpu / gpu
# Use Device# {{{
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
if use_device == 'gpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
elif use_device == 'cpu':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# }}}
# import# {{{
import pandas as pd
import numpy as np
import sys
# }}}
# import keras# {{{
import keras.backend as K
from keras.models import Sequential, load_model
from keras.models import Model
from keras.layers import Input, Flatten, Embedding, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add, Dot, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
# }}}
def RMSE(y_true, y_pred):# {{{
    return K.sqrt(K.mean(K.square(y_pred - y_true)))
# }}}
ID = 35
# get movie embedding# {{{
model = load_model('./model/{}.h5'.format(ID), custom_objects={'RMSE': RMSE})
# model = load_model('./model/daniel_model.h5', custom_objects={'rmse': RMSE})
# model.summary()

movie_emb = np.array(model.layers[3].get_weights()).squeeze()
print('movie embedding shape:', movie_emb.shape)
np.save('movie_emb.npy', movie_emb)
# movie_emb = np.load('movie_emb.npy')
# }}}
# load genres from movies.csv# {{{
movie_path = './data/movies.csv'
genres = list(pd.read_csv(movie_path, delimiter='::')['Genres'])
genres = [g.split('|') for g in genres]
movie_id = list(pd.read_csv(movie_path, delimiter='::')['movieID'] - 1)
# }}}
# generate category number# {{{
cnum = np.zeros(3952, dtype=int)
for i, g in enumerate(genres):
    has_tag = []
    if 'Drama' in g or 'Musical' in g:
        has_tag += [1]
    elif 'Thriller' in g or 'Horror' in g or 'Crime' in g:
        has_tag += [2]
    elif 'Adventure' in g or 'Animation' in g or "Children's" in g:
        has_tag += [3]

    if len(has_tag) == 0:
        cnum[movie_id[i]] = 0
    else:
        cnum[movie_id[i]] = has_tag[ np.random.randint(len(has_tag)) ]
# }}}
# draw# {{{
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# y = np.array(y)
# x = np.array(x, dtype=np.float64)

new_emb = [movie_emb[i] for i,c in enumerate(cnum) if c != 0]
new_c   = [c for i,c in enumerate(cnum) if c != 0]

# perform t-SNE embedding
model = TSNE(n_components=2, random_state=0)
print('start fit')
np.set_printoptions(suppress=True)
vis_data = model.fit_transform(new_emb)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

cm = plt.cm.get_cmap('RdYlBu')
sc = plt.scatter(vis_x, vis_y, c=new_c, l=new_c, cmap=cm)
plt.legend(loc='upper right')
# plt.colorbar(sc)
plt.show()
# }}}
