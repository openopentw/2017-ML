from keras import backend as K

THRES = 0.33

def f1score(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)

    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

def thres_max_f1score(y_true, y_pred):
    y_pred_max = K.max(y_pred, axis=1)
    y_thres = K.reshape(y_pred_max * 0.33, (K.shape(y_pred)[0], 1))

    y_prob = K.cast(K.greater(y_pred, y_thres), K.floatx())
    return f1score(y_true, y_prob)

def thres_f1score(y_true,y_pred):
    y_pred = K.cast(K.greater(y_pred, THRES), dtype='float32')
    tp = K.sum(y_true * y_pred)

    precision=tp/(K.sum(y_pred))
    recall=tp/(K.sum(y_true))
    return 2*((precision*recall)/(precision+recall))
