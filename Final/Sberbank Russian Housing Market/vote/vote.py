import numpy as np

ID = 1
output_path = '../subm/vote_{}.csv'.format(ID)
subm_list = [
    '../subm/submission_3.csv',
    '../subm/submission_4.csv',
    '../subm/submission_5.csv'
]

subm = np.zeros((len(subm_list), 7662))
for i, s in enumerate(subm_list):
    subm[i] = np.genfromtxt(s, delimiter=',')[1:, 1]

y_pred = np.round(np.mean(subm, axis=0)).astype(int)

def save_submission(y_pred):# {{{
    f = open(output_path, 'w')
    print('id,price_doc', file=f)
    for i in range(y_pred.size):
        print(str(30474+i) + ',' + str(int(y_pred[i])), file=f)
    f.close()
save_submission(y_pred)# }}}
