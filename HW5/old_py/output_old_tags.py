#! python3
"""
@author: b04902053
"""

import numpy as np

# Argvs
TRAIN_FILE = './data/new_train_data.csv'
TEST_FILE = './data/test_data.csv'
OUTPUT = './submission.csv'

# Load dataset
print('Processing text dataset')

all_tags = []
tags = []
txts = []

f = open(TRAIN_FILE, 'r', encoding='utf8')
lines = f.readlines()[1:]
f.close()
for i in range(len(lines)):
    lines[i] = lines[i].split('"')[1:]
    for rest in lines[i][2:]:
        lines[i][1] += rest

    line_tag = lines[i][0].split(' ')
    all_tags += line_tag
    tags += [line_tag]
    txts += [ lines[i][1] ]

# y_train_data
all_tags = set(all_tags)
tag2int = {tag:i for i, tag in enumerate(all_tags)}
int2tag = {i:tag for i, tag in enumerate(all_tags)}

# print tags
f = open('old_tags', 'w')
for i in range(len(int2tag)):
    a += [int2tag[i]]
    print(a[i], file=f)
f.close()
