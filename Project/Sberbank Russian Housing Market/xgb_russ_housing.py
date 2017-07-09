from random import randint
import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime
#now = datetime.datetime.now()

ID = 32
# argv# {{{
train_path = './data/train.csv'
test_path  = './data/test.csv'
macro_path = './data/macro.csv'

output_path = './subm/submission_{}.csv'.format(ID)
print('Will save subm.csv to: {}'.format(output_path))
# }}}

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
macro = pd.read_csv(macro_path)
id_test = test.id

y_train = train["price_doc"] * .95 + 10.05
x_train = train.drop(["id", "timestamp", "price_doc"], axis=1)
x_test = test.drop(["id", "timestamp"], axis=1)

#can't merge train with test because the kernel run for very long time

for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values)) 
        x_train[c] = lbl.transform(list(x_train[c].values))
        #x_train.drop(c,axis=1,inplace=True)

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
        #x_test.drop(c,axis=1,inplace=True)    


xgb_params = {
    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=1000, early_stopping_rounds=20,
    verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round= num_boost_rounds)


y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})

print('Saving subm.csv to: {}'.format(output_path))
output.to_csv(output_path, index=False)
