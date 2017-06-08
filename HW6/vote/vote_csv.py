"""
@author: b04902053
"""
# import# {{{
import pandas as pd
import numpy as np
# }}}
# Parameter #
ID = 14
ADD_NUM = 0.07
# argvs# {{{
print('ID = {}'.format(ID))

output_path = '../subm/vote_{}.csv'.format(ID)
print('Will save output to: {}'.format(output_path))
# }}}
# subm list #
subm_list = [
    '../subm/submission_35.csv',
    '../subm/submission_36.csv',
    # '../subm/submission_37.csv',
    # '../subm/submission_40.csv',
    # '../subm/submission_41.csv',
    # '../subm/submission_42.csv',
    # '../subm/submission_43.csv',
    # '../subm/submission_44.csv',
    # '../subm/submission_45.csv',
    # '../subm/submission_46.csv',
    # '../subm/submission_47.csv',
    # '../subm/submission_48.csv',
    '../subm/submission_49.csv',
]
score = np.array([# {{{
    3987,
    3993,
    # 4197,
    4039,
    # 4149,
    4071,
    4004,
    4077,
    4058,
    4080,
    4115,
    4031,
    3997,
])# }}}
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
# y_pred = np.average(preds, axis=0, weights=10000/score)
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
