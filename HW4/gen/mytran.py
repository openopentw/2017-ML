import numpy as np

# file_name = 'mm_x_train'
file_name = 'mm_x_vali'
INFILE = file_name + '.csv'
OUTFILE = file_name + '.npy'

raw = np.genfromtxt(INFILE, delimiter=',')
# raw[:,:100] /= raw[:,100].reshape(raw.shape[0], 1)
raw = raw[:,40:100]

np.save(OUTFILE, raw)
