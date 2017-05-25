#! python3
"""
@author: b04902053
"""

import numpy as np

THRES = 0.33

csv_list = [
    # './1.csv',
    # './2.csv',
    './3.csv',
    # './4.csv',
    './5.csv',
    './400_prob.csv'
    ]

probs = np.zeros((len(csv_list), 1234, 38))
for i, c in enumerate(csv_list):
    probs[i] = np.genfromtxt(c)

# threshold# {{{
def threshold(y_prob):
    y_prob_max = np.max(y_prob, axis=1).reshape(y_prob.shape[0], 1)
    y_prob[y_prob == y_prob_max] = 1

    y_prob[y_prob > THRES] = 1
    y_prob[y_prob != 1] = 0
    return y_prob
# }}}

for i in range(probs.shape[0]):
    probs[i] = threshold(probs[i])

# linear combination# {{{
y_prob = np.sum(probs, axis=0) / 3
y_prob[y_prob > 1.5/3] = 1
y_prob[y_prob != 1] = 0
# }}}

# change to tags# {{{
y_tags = []
for i in range(y_prob.shape[0]):
    tag = []
    for j in range(38):
        if y_prob[i][j] == 1:
            tag += [int2tag[j]]
    y_tags += [tag]
# }}}

# save to subm.csv# {{{
f = open('submission.csv', 'w')
print('"id","tags"', file=f)
for i, ts in enumerate(y_tags):
    print('"{}","'.format(i), end='', file=f)
    for i, t in enumerate(ts):
        if i != 0:
            print(' ', end='', file=f)
        print(t, end='', file=f)
    print('"', file=f)
f.close()
# }}}
