import numpy as np

VALI_THRES = 0.33

def np_f1score(y_true, y_pred):
    num_tp = np.sum(y_true*y_pred)
    num_fn = np.sum(y_true*(1.0-y_pred))
    num_fp = np.sum((1.0-y_true)*y_pred)

    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def np_thres_f1score(y_true, y_prob):
    y_prob_max = np.max(y_prob, axis=1).reshape(y_prob.shape[0], 1)
    y_prob[y_prob == y_prob_max] = 1

    y_prob[y_prob > VALI_THRES] = 1
    y_prob[y_prob != 1] = 0

    np.savetxt('vali.csv', y_prob)

    return np_f1score(y_true, y_prob)

def np_max_thres_f1score(y_true, y_prob):
    y_prob_max = np.max(y_prob, axis=1).reshape(y_prob.shape[0], 1)
    y_thres = y_prob_max * VALI_THRES

    y_prob[y_prob > y_thres] = 1
    y_prob[y_prob != 1] = 0

    return np_f1score(y_true, y_prob)

y_prob = model.predict(x_vali)
f1 = np_thres_f1score(y_vali, y_prob)

print('vali:')
print(f1)
