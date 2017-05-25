# import# {{{
import pandas as pd
import numpy as np
# keras# {{{
from keras import backend as K
from keras.models import Sequential, load_model
from keras.callbacks import  ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint, EarlyStopping
# }}}
# }}}
# parameters #
ID = 9
EPOCHS = 250
PATIENCE = 30

BATCH_SIZE = 128
VALI = -3000
# argv# {{{
train_path  = './data/train.csv'
test_path   = './data/test.csv'
output_path = './subm/submission_{}.csv'.format(ID)

model_path  = './model/{}.h5'.format(ID)
weights_path = './best_weights_{}.h5'.format(ID)
# }}}

# Preprocessing #
def read_data(data_path): # load train & convert to numpy# {{{
    print('loading data')
    data = pd.read_csv(data_path)
    data.fillna(0, inplace=True)
    return data
train_data = read_data(train_path)
test_data  = read_data(test_path)
# }}}
def choose_data(data):# {{{
    # data = data[['full_sq', 'life_sq', 'floor', 'max_floor', 'num_room', 'area_m', 'raion_popul', 'young_all', 'young_female',
                            # 'work_all', 'work_male', 'work_female', 'preschool_quota', 'university_top_20_raion', 'university_km']]
    drop_list = ['id', 'timestamp', 'max_floor', 'material', 'build_year', 'num_room', 'kitch_sq', 'state', 'sub_area',
            'culture_objects_top_25', 'cafe_sum_500_min_price_avg', 'cafe_sum_500_max_price_avg', 'cafe_avg_price_500']
    for s in drop_list:
        data.drop(s, axis=1, inplace=True)
    if 'price_doc' in data:
        data.drop('price_doc', axis=1, inplace=True)

    data = data.values
    return data
y_train_data = train_data['price_doc'].values
x_train_data = choose_data(train_data)
test_data    = choose_data(test_data)
# }}}
print('preprocessing data')
def all_to_number(data): # change string to number# {{{
    string_to_int = {
        'OwnerOccupier': 0, 'Investment': 1,
        'no': 0, 'yes': 1,
        'excellent': 4, 'good': 3, 'satisfactory': 2, 'poor': 1, 'no data': 2
    }
    for k, v in string_to_int.items():
        data[data == k] = v
    data = data.astype(float)
    return data
x_train_data = all_to_number(x_train_data)
test_data  = all_to_number(test_data)
# }}}
# TODO: change sub_area to vector
# split x & y# {{{
# x_train_data = train_data[:,:-1]
# y_train_data = train_data[:,-1]
# }}}
# normalization# {{{
mean = np.mean(x_train_data, axis=0).reshape(1, x_train_data.shape[1])
std = np.std(x_train_data, axis=0).reshape(1, x_train_data.shape[1])
x_train_data = (x_train_data - mean) / std
test_data  = (test_data - mean) / std
# }}}

# split train & vali# {{{
x_vali = x_train_data[VALI:]
y_vali = y_train_data[VALI:]
x_train = x_train_data[:VALI]
y_train = y_train_data[:VALI]
x_test = test_data
# }}}
def RMSLE(y_true, y_pred):# {{{
    y_pred_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    y_true_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(y_pred_log - y_true_log), axis = -1))
# }}}
def generate_model(input_dim):# {{{
    model = Sequential()

    # model.add(Conv1D(32, 3, activation='relu', input_shape=(48, 48, 1)))
    # model.add(Dropout(0.3))

    # model.add(Flatten(input_shape=input_shape))
    # model.add(Flatten())

    model.add(Dense(output_dim=512, activation='elu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(1024, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='elu'))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='elu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.summary()

    return model
model = generate_model(x_train.shape[1])
# }}}
# compile & fit# {{{
earlystopping = EarlyStopping(monitor='val_mean_squared_logarithmic_error', patience=PATIENCE, verbose=1, mode='min')
checkpoint = ModelCheckpoint(filepath=weights_path, verbose=1, save_best_only=True, save_weights_only=True, monitor='val_mean_squared_logarithmic_error', mode='min')

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['msle'])
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_vali, y_vali), callbacks=[earlystopping, checkpoint])
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['msle'])
# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=50, validation_data=(x_vali, y_vali), callbacks=[earlystopping, checkpoint])
# model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['msle'])
# model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS-50, validation_data=(x_vali, y_vali), callbacks=[earlystopping, checkpoint])
# }}}
# load best & predict# {{{
print('Loading best model: {}'.format(weights_path))
model.load_weights(weights_path)
model.save(model_path)

score = model.evaluate(x_train, y_train)
print ('Train Acc:', score)
score = model.evaluate(x_vali, y_vali)
print ('Vali Acc:', score)
y_pred = model.predict(x_test)
y_print = y_pred
# }}}
def save_submission(y_pred):# {{{
    f = open(output_path, 'w')
    print('id,price_doc', file=f)
    for i in range(y_pred.size):
        print(str(30474+i) + ',' + str(int(y_pred[i][0])), file=f)
    f.close()
save_submission(y_print)# }}}
