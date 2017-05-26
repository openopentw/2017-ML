import numpy as np
import gen

thres = np.zeros(60)
for i in range(1, 60):
    dim = i

    for j in range(1, 11):
        N = 10000 * j
        for k in range(2):
            # gen data
            layer_dims = [np.random.randint(60, 80), 100]

            data = gen.gen_data(dim, layer_dims, N)
            # (data, dim) is a (question, answer) pair

            # pca
            val, vec = np.linalg.eigh(np.cov(data.T))
            val = val[::-1] / val.sum()

            before = np.sum(val[:dim])
            thres[i] += before
    thres[i] /= 100
    print(i, thres[i], sep=',')
thres = thres[1:]
