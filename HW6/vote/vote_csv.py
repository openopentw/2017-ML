"""
@author: b04902053
"""
# import# {{{
import pandas as pd
import numpy as np
# }}}
# Parameter #
ID = 3
ADD_NUM = 0.05
ROUND_DIFF = 0.11
# argvs# {{{
print('ID = {}'.format(ID))

output_path = '../subm/vote_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))
# }}}
# subm list #
subm_list = [
    '../subm/submission_35.csv',
    '../subm/submission_36.csv',
]
# read csvs# {{{
preds = np.zeros((len(subm_list), 100336))
for i, subm in enumerate(subm_list):
    print('loading csv from {}'.format(subm))
    preds[i] = pd.read_csv(subm)['Rating'].values
# }}}
# mean & add something# {{{
y_pred = np.mean(preds, axis=0)
y_pred += ADD_NUM
# }}}
# special rounds# {{{
y_pred[y_pred < 1] = 1
y_pred[y_pred > 5] = 5

round_pred = np.round(y_pred)
diff_pred = np.abs(round_pred - y_pred)
y_pred[diff_pred < ROUND_DIFF] = np.round( y_pred[diff_pred < ROUND_DIFF] )
# }}}
# save to h5 & csv# {{{
print('Saving submission to {}'.format(output_path))
f = open(output_path, 'w')
print('TestDataID,Rating', file=f)
for i, pred_rate in enumerate(y_pred):
    print('{},{}'.format(i+1, pred_rate), file=f)
f.close()
# }}}
