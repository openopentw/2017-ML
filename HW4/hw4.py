import numpy as np
from sklearn import decomposition

def deelu(x):
    # return where(x>=0, x, np.log(x+1))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                x[i][j] = np.log(x[i][j]+1+1e-4)
    return x

# data = np.load('./data.npz')

# for i in range(200):
x = data['0']
pca = decomposition.PCA(n_components=100).fit_transform(x)
"""
pca = deelu(pca)
pca = decomposition.PCA(n_components=70).fit_transform(x)
pca = deelu(pca)
val, vec = np.linalg.eigh(np.cov(pca.T))
a = 0
for j in range(69, 0, -1):
    a += val[j]
    if a/val.sum() > 0.9:
        print(70 - j)
        break
"""
