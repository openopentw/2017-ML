import numpy as np
import word2vec

"""
word2vec
"""
word2vec.word2phrase('./all.txt', './all-phrases', verbose=True)

# train model
word2vec.word2vec(
        train = './all-phrases',
        output = './all.bin',
        # size=200,
        window=25,
        negative=5,
        # # iter_=ITERATIONS,
        min_count=100,
        # alpha=0.025,
        # cbow=1,
        threads=4
        )
