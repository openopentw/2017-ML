#! python3
"""
@author: b04902053
"""

import numpy as np

data = np.load('d:/data.npz')

eigh = np.zeros((200, 60))
for i in range(200):
    x = data[str(i)]
    val, vec = np.linalg.eigh(np.cov(x.T))
    val = val[::-1] / val.sum()
    eigh[i] = val[:60]

np.savetxt('./eigh.csv', eigh)
