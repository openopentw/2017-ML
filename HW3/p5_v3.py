import os
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
from utils import *
from marcos import *
import numpy as np

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-7)

def grad_ascent(num_step,input_image_data,iter_func):
    """
    Implement this function!
    """
    return filter_images

emotion_classifier = load_model(model_path)
layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers[1:])
input_img = emotion_classifier.input

name_ls = ["names of the layers you want to get their outputs"]
collect_layers = [ layer_dict[name].output for name in name_ls ]

for cnt, c in enumerate(collect_layers):
    filter_imgs = [[] for i in range(NUM_STEPS//RECORD_FREQ)]
    for filter_idx in range(nb_filter):
        input_img_data = np.random.random((1, 48, 48, 1)) # random noise
        target = K.mean(c[:, :, :, filter_idx])
        grads = normalize(K.gradients(target, input_img)[0])
        iterate = K.function([input_img], [target, grads])

        ###
        "You need to implement it."
        filter_imgs = grad_ascent(num_step, input_img_data, iterate)
        ###

    for it in range(NUM_STEPS//RECORD_FREQ):
        fig = plt.figure(figsize=(14, 8))
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/16, 16, i+1)
            ax.imshow(filter_imgs[it][i][0], cmap='BuGn')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('{:.3f}'.format(filter_imgs[it][i][1]))
            plt.tight_layout()
        fig.suptitle('Filters of layer {} (# Ascent Epoch {} )'.format(name_ls[cnt], it*RECORD_FREQ))
        img_path = os.path.join(filter_dir, '{}-{}'.format(store_path, name_ls[cnt]))
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path,'e{}'.format(it*RECORD_FREQ)))

