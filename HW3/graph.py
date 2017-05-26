import numpy as np
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Input, Conv2D, concatenate
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import np_utils
import sys







def load_data():
	
	f = open( sys.argv[1], 'r' )
	data = []
	for line in f.readlines():
		data.append(line.replace(',', ' ').split())
	data = np.array(data[1:]).astype('float32')

	x_train = data[:, 1:]
	x_train /= 255
	y_train = data[:, 0]
	y_train = np_utils.to_categorical(y_train, 7)

	f = open( sys.argv[2], 'r' )
	data = []
	for line in f.readlines():
		data.append(line.replace(',', ' ').split())
	data = np.array(data[1:]).astype('float32')

	x_test = data[:, 1:]
	x_test /= 255

	return (x_train, y_train), (x_test)


(x_train, y_train), (x_test) = load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
x_train = x_train.reshape( (28709, 48, 48, 1) )		# 28709 = 19×1511
x_test = x_test.reshape( (7178, 48, 48, 1) )



input_img = Input( shape = (48, 48, 1))

tower_1 = Conv2D(32, (3, 3), padding='same', activation='elu')(input_img)
tower_1 = Conv2D(32, (3, 3), padding='same', activation='elu')(tower_1)
tower_1 = Flatten()(tower_1)

tower_2 = Conv2D(32, (5, 5), padding='same', activation='elu')(input_img)
tower_2 = Conv2D(32, (5, 5), padding='same', activation='elu')(tower_2)
tower_2 = Flatten()(tower_2)

tower_3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_img)
tower_3 = Conv2D(32, (3, 3), padding='same', activation='elu')(tower_3)
tower_3 = Flatten()(tower_3)

concat = concatenate([tower_1, tower_2, tower_3])

drop_1 = Dropout(0.25)(concat)
den_1 = Dense(64, activation='elu')(drop_1)
drop_2 = Dropout(0.5)(den_1)
den_2 = Dense(32, activation='elu')(drop_2)
drop_3 = Dropout(0.5)(den_2)

output = Dense(7, activation='softmax')(drop_3)

model = Model(input=[input_img], output=output)
# model.add( Dropout(0.25) )

# model.add( Dense(output_dim = 128) )
# model.add( Activation('elu') )
# model.add( Dropout(0.5) )
# model.add( Dense(output_dim = 32) )
# model.add( Activation('elu') )
# model.add( Dropout(0.5) )
# model.add( Dense(output_dim = 7) )
# model.add( Activation('softmax') )

model.compile( loss = 'categorical_crossentropy', optimizer = 'Adadelta', metrics=['accuracy'] )

model.summary()
model.fit( x_train, y_train, batch_size = 100, epochs = 20, validation_split = 0.2)

model.save( sys.argv[3] )

score = model.evaluate(x_train, y_train)
print("\nacc :", score[1])

y_test = model.predict(x_test)

f = open("out.csv", 'w')
print("label,id", file = f)
for i in range(7178):
	ans = 0
	for k in range(7):
		if y_test[i][k] > y_test[i][ans]:
			ans = k

	print(i, ",", ans, sep = '', file = f)



