from scipy.misc import imread
import numpy as np

choose = ['A','B','C','D','E',
          'F','G','H','I','J']

img = np.zeros((10, 75, 64, 64))
for i, c in enumerate(choose):
    print(c)
    for j in range(0, 75):
        # print(c + "{0:0=2d}".format(i))
        # img[i][j] = imread('./data/' + c + "{0:0=2d}".format(j) + '.bmp')
        img[i][j] = imread('./data/{}{:02d}.bmp'.format(c, j))
img = img.reshape(10, 75, 64*64)
img = img - np.average(img, axis=2).reshape(10, 75, 1)

# eig = np.zeros((10, 75, 75))
# for i in range(10):
X = img[0]
U, s, Vt = np.linalg.svd(X)
V = Vt.T
S = np.diag(s)
Mhat = np.dot(U[:,:9], np.dot(S[:9,:9], V[:,:9].T))

# eig[i] = np.dot(U, np.dot(S, V))
