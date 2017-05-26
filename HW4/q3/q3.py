#! python3
"""
@author: b04902053
"""

import numpy as np
from scipy.misc import imread

NDATA = 481

data = np.zeros((NDATA, 480, 512))
for i in range(1, NDATA+1):
    data[i-1] = imread('./hand/hand.seq{}.png'.format(i))
data = data.reshape(NDATA, 480*512)

# val, vec = np.linalg.eigh(np.cov(data.T))
val, vec = np.linalg.eigh(np.dot(data, data.T))
eig = val[:60].reshape(1, 60) / 10000

np.savetxt('./hand_eigh.csv', eig)
