import numpy as np

ID = 9
output_path = '../subm/vote_{}.csv'.format(ID)
subm_list = [
    '../subm/submission_12.csv',
    '../subm/submission_25.csv',
    '../subm/submission_26.csv',
]

subm = np.zeros((len(subm_list), 7662))
for i, s in enumerate(subm_list):
    subm[i] = np.genfromtxt(s, delimiter=',')[1:, 1]

y_pred  = np.round(np.mean(subm, axis=0))
y_pred *= 0.97
y_pred = y_pred.astype(int)

def save_submission(y_pred):# {{{
    f = open(output_path, 'w')
    print('id,price_doc', file=f)
    for i in range(y_pred.size):
        print(str(30474+i) + ',' + str(int(y_pred[i])), file=f)
    f.close()
save_submission(y_pred)# }}}
