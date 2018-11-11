import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import utils

np.random.seed(42)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


# Normalize import date
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

model = Sequential()

# layers
model.add(Convolution2D(32, (3, 3), border_mode='same', input_shape=(3, 32, 32), activation='relu'))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2D to 1D
model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


model.compile(loss='categorial_crossentropy', optimizer='SGD', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, nb_epoch=25, validation_split=0.1, shuffle=True)

