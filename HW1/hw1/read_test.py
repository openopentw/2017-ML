TEST = np.genfromtxt('data/test_X.csv', delimiter=',')
TEST = TEST[:,2:]

# convert test to float
test = np.nan_to_num(test)
test = test.astype(float)

TEST = TEST.reshape(240, 18, 9)
TEST = (TEST - MEAN) / VAR
TEST = TEST[:, FEATURE,:]

# append TEST & DIMS together
TEST = TEST.reshape(240*N_FEATURE, 9)
TEST_DIM = np.zeros((TEST.shape[0], TEST.shape[1] * DIM))
for i in range(DIM):
    TEST_DIM[:, 9*i: 9*(i+1)] = TEST ** (i+1)
TEST_DIM = TEST_DIM.reshape(240, int(TEST_DIM.size / 240))

ans = b + np.dot(TEST_DIM , W.T)
