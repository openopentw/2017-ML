import numpy as np
import gen

NUM_ARRAY = [1, 2, 5, 8, 10]
NUM_PER_DIM = 20

x_train = np.zeros((60, 5, NUM_PER_DIM, 101))
y_train = np.zeros((60, 5, NUM_PER_DIM, 1))
for i in range(60):
    dim = i+1
    print(dim)

    for j in range(5):
        print('\t', NUM_ARRAY[j], end='')
        N = 10000 * NUM_ARRAY[j]
        for k in range(NUM_PER_DIM):
            # gen data
            layer_dims = [np.random.randint(60, 80), 100]

            # (data, dim) is a (question, answer) pair
            data = gen.gen_data(dim, layer_dims, N)

            # pca
            val, vec = np.linalg.eigh(np.cov(data.T))
            val = val[::-1]

            x_train[i][j][k][:-1] = val
            x_train[i][j][k][-1] = val.sum()
            y_train[i][j][k][0] = dim
    print('')

x_train = x_train.reshape(300 * NUM_PER_DIM, 101)
np.savetxt('50_x_train.csv', x_train, delimiter=',')

y_train = y_train.reshape(300 * NUM_PER_DIM, 1)
np.savetxt('50_y_train.csv', y_train, delimiter=',')
