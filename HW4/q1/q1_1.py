from scipy.misc import imread, imsave
import numpy as np
import matplotlib.pyplot as plt
import os

# choose image
CHOOSE = ['A','B','C','D','E',
          'F','G','H','I','J']
NUM = 10

# input & reshape to 2d
img = np.zeros((10, NUM, 64, 64))
for i, c in enumerate(CHOOSE):
    print(c)
    for j in range(0, 10):
        img[i][j] = imread('./data/{}{:02d}.bmp'.format(c, j))
img = img.reshape(10*NUM, 64*64)

# average_face
avg_img = np.average(img, axis=0).reshape(64, 64)
imsave(os.path.join('./', 'q1_1_avg.jpg'), avg_img)

# pca
img = img - np.average(img, axis=1).reshape(10*NUM, 1)
U, s, V = np.linalg.svd(img)
eig = V[:9].reshape(9, 64, 64)

# print pca to 3-by-3 image
fig = plt.figure(figsize=(3, 3))
for i, img in enumerate(eig):
    ax = fig.add_subplot(eig.shape[0]/3, 3, i+1)
    ax.imshow(img, cmap='gray')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    # plt.tight_layout()
fig.suptitle('Top 9 Eigenfaces')
fig.savefig(os.path.join('./', 'q1_1_eig.jpg'))
