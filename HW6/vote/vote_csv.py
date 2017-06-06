"""
@author: b04902053
"""
# import# {{{
import pandas as pd
import numpy as np
# }}}
# Parameter #
ID = 6
ADD_NUM = 0.05
# argvs# {{{
print('ID = {}'.format(ID))

output_path = '../subm/vote_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))
# }}}
# subm list #
subm_list = [
    '../subm/submission_17.csv',
    '../subm/submission_34.csv',
    '../subm/submission_35.csv',
    '../subm/submission_36.csv',
    '../subm/submission_37.csv',
    '../subm/submission_40.csv',
]
# read csvs# {{{
print('')
preds = np.zeros((len(subm_list), 100336))
for i, subm in enumerate(subm_list):
    print('loading csv from {}'.format(subm))
    preds[i] = pd.read_csv(subm)['Rating'].values
print('')
# }}}
# mean & add something# {{{
y_pred = np.mean(preds, axis=0)
y_pred += ADD_NUM
# }}}
# clip on 1 & 5# {{{
y_pred[y_pred < 1] = 1
y_pred[y_pred > 5] = 5
# }}}
# save to h5 & csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, pred_rate), file=f)
f.close()
# }}}
