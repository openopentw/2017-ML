"""
@author: b04902053
"""
# import# {{{
import pandas as pd
import numpy as np
# }}}
# Parameter #
ID = 16
ADD_NUM = 0.07
# argvs# {{{
print('ID = {}'.format(ID))

output_path = '../subm/vote_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))
# }}}
# subm list #
subm_list = [
    '../subm/submission_6.csv',
    '../subm/submission_7.csv',
    '../subm/submission_8.csv',
    '../subm/submission_9.csv',
    '../subm/submission_10.csv',
    '../subm/submission_12.csv',
    '../subm/submission_13.csv',
    '../subm/submission_23.csv',
    '../subm/submission_24.csv',
    '../subm/submission_26.csv',
    '../subm/submission_27.csv',
    '../subm/submission_28.csv',
    '../subm/submission_29.csv',
    '../subm/submission_30.csv',
    '../subm/submission_31.csv',
    '../subm/submission_32.csv',
    '../subm/submission_33.csv',
    '../subm/submission_38.csv',
    '../subm/submission_39.csv',

    '../subm/submission_11.csv',
    '../subm/submission_14.csv',
    '../subm/submission_15.csv',
    '../subm/submission_16.csv',
    '../subm/submission_17.csv',
    '../subm/submission_18.csv',
    '../subm/submission_19.csv',
    '../subm/submission_20.csv',
    '../subm/submission_21.csv',
    '../subm/submission_22.csv',
    '../subm/submission_25.csv',
    '../subm/submission_34.csv',
    '../subm/submission_35.csv',
    '../subm/submission_36.csv',
    '../subm/submission_37.csv',
    '../subm/submission_40.csv',
    '../subm/submission_41.csv',
    '../subm/submission_42.csv',
    '../subm/submission_43.csv',
    '../subm/submission_44.csv',
    '../subm/submission_45.csv',
    '../subm/submission_46.csv',
    '../subm/submission_47.csv',
    '../subm/submission_48.csv',
    '../subm/submission_49.csv',
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
