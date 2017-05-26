import numpy as np
from sklearn import decomposition

data = np.load('./data.npz')

ans = np.zeros(200)
for i in range(200):
    x = data[str(i)]
    pca = decomposition.PCA(n_components=100).fit_transform(x)
    pca = decomposition.PCA(n_components=70).fit_transform(x)
    val, vec = np.linalg.eigh(np.cov(pca.T))
    dim = 0
    for j in range(69, 0, -1):
        dim += val[j]
        if dim/val.sum() > 0.85:
            ans[i] = np.log(70-j)
            print("{},{}".format(i, 70-j))
            break

f = open("./submission.csv", "w")
print("SetId,LogDim", file=f)
for i in range(200):
    print("{},{}".format(i, ans[i]), file=f)
f.close()
