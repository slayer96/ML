import numpy as np

from keras.datasets import boston_housing
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(42)

# load data
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()

# data standardization
mean = x_train.mean(axis=0)
std = x_train.std(axis=0)
x_train -= mean
x_train /= std
x_test -= mean
x_test /= std

# create model
model = Sequential()

# add layers
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse', metrics=['mae'])


# train network
model.fit(x_train, y_train, epochs=250, batch_size=1, verbose=1)


# learning quality assessment
mse, mae = model.evaluate(x_test, y_test, verbose=0)
print(mse)

# print test results
pred = model.predict(x_test)
print('test 1: ', pred[1][0], y_test[1])

print('test 2: ', pred[100][0], y_test[100])