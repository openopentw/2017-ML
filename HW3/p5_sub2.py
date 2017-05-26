from scipy.misc import imread
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
# from utils import *
import numpy as np

choose_id = 678
model = load_model('./674561.hdf5')
layer_dict = dict([layer.name, layer] for layer in model.layers[1:])

input_img = model.input

name_ls = ['conv2d_2']
collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]

photo = imread('./pngs/train/678.png').reshape(1, 48, 48, 1)
for cnt, fn in enumerate(collect_layers):
    im = fn([photo, 0]) #get the output of that layer
    fig = plt.figure(figsize=(14, 8))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/8, 8, i+1)
        ax.imshow(im[0][0, :, :, i], cmap='BuGn')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
    fig.savefig('p5_sub2')
