#! python3
"""
@author: b04902053
"""

import numpy as np

data = np.load('./data.npz')
thres = np.genfromtxt('./gen/thres.csv', delimiter=',')[:,1]

ans = np.zeros(200)
ans += np.log(60)
for i in range(200):
    x = data[str(i)]
    val, vec = np.linalg.eigh(np.cov(x.T))
    val = val[::-1] / val.sum()
    tmp_sum = 0
    for j in range(59):
        tmp_sum += val[j]
        if tmp_sum > thres[j]:
            ans[i] = np.log(j)
            print("{},{}".format(i, j))
            break

f = open("./submission.csv", "w")
print("SetId,LogDim", file=f)
for i in range(200):
    print("{},{}".format(i, ans[i]), file=f)
f.close()
