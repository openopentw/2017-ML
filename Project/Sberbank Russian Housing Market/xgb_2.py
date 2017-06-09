# import# {{{
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
# }}}
# parameters #
ID = 22
nrounds = 1000
patience = 20
xgb_params = {# {{{
    'objective': 'reg:linear',

    'eta': 0.05,
    'max_depth': 5,
    'subsample': 0.7,

    'colsample_bytree': 0.7,
    'eval_metric': 'rmse',
    'silent': 1,
}# }}}

# argv# {{{
train_path = './data/train.csv'
test_path  = './data/test.csv'
macro_path = './data/macro.csv'

output_path = './subm/submission_{}.csv'.format(ID)
print('Will save subm.csv to: {}'.format(output_path))
# }}}

# load data# {{{
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
id_test = test.id

'''
macro_cols = ["balance_trade", "balance_trade_growth", "eurrub", "average_provision_of_build_contract",
                "micex_rgbi_tr", "micex_cbi_tr", "deposits_rate", "mortgage_value", "mortgage_rate",
                "income_per_cap", "rent_price_4+room_bus", "museum_visitis_per_100_cap", "apartment_build"]
macro = pd.read_csv(macro_path, parse_dates=['timestamp'], usecols=['timestamp'] + macro_cols)
'''
# }}}
def choose_data(data):# {{{
    drop_list = ['id']
    # drop_list = ['id', 'max_floor', 'material', 'build_year', 'num_room', 'kitch_sq', 'state', 'culture_objects_top_25']
            # 'culture_objects_top_25', 'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500']
    for s in drop_list:
        data.drop(s, axis=1, inplace=True)
    return data
train = choose_data(train)
test  = choose_data(test)
# }}}
# split y_train# {{{
y_train = train["price_doc"]
x_train = train.drop(["price_doc"], axis=1)
x_test  = test
# }}}
# let labels be int# {{{
for c in x_train.columns:
    if x_train[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_train[c].values))
        x_train[c] = lbl.transform(list(x_train[c].values))

for c in x_test.columns:
    if x_test[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(x_test[c].values)) 
        x_test[c] = lbl.transform(list(x_test[c].values))
# }}}
# convert to DMatrix# {{{
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
# }}}

cv_output = xgb.cv(xgb_params, dtrain, num_boost_round=nrounds, early_stopping_rounds=patience, verbose_eval=50, show_stdv=False)
cv_output[['train-rmse-mean', 'test-rmse-mean']].plot()

num_boost_rounds = len(cv_output)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round= num_boost_rounds)

y_predict = model.predict(dtest)
output = pd.DataFrame({'id': id_test, 'price_doc': y_predict})
output.head()

print('Saving subm.csv to: {}'.format(output_path))
output.to_csv(output_path, index=False)
