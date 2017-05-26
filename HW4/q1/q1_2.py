from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os

# choose image
CHOOSE = ['A','B','C','D','E',
          'F','G','H','I','J']
NUM = 10
K = 5

# input & reshape to 2d
img = np.zeros((10, NUM, 64, 64))
for i, c in enumerate(CHOOSE):
    print(c)
    for j in range(0, 10):
        img[i][j] = imread('./data/{}{:02d}.bmp'.format(c, j))
img = img.reshape(10*NUM, 64, 64)

# print 10-by-10 original img
fig = plt.figure(figsize=(10, 10))
for i, m in enumerate(img):
    ax = fig.add_subplot(img.shape[0]/10, 10, i+1)
    ax.imshow(m, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fig.suptitle('Original faces')
fig.savefig(os.path.join('./', 'q1_2_ori.jpg'))

# pca
img = img.reshape(10*NUM, 64*64)
avg_img = np.average(img, axis=1).reshape(10*NUM, 1)
img = img - np.average(img, axis=1).reshape(10*NUM, 1)
U, s, V = np.linalg.svd(img)

# Pca Reconstruct
S = np.diag(s)
recon = np.dot(U[:,:K], np.dot(S[:K,:K], V[:K,:]))
recon += avg_img
recon = recon.reshape(10*NUM, 64, 64)

# print pca to 10-by-10 image
fig = plt.figure(figsize=(10, 10))
for i, m in enumerate(recon):
    ax = fig.add_subplot(recon.shape[0]/10, 10, i+1)
    ax.imshow(m, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
fig.suptitle('Reconstruct faces')
fig.savefig(os.path.join('./', 'q1_2_recon.jpg'))
