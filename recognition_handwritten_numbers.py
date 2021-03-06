import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.datasets import mnist

np.random.seed(42)

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#  transformation data
X_train = X_train.reshape(60000, 784)
# normalisation data
X_train = X_train.astype('float32')
X_train /= 255

# transformation labels to categories
y_train = np_utils.to_categorical(y_train, 10)

# create model
model = Sequential()

# add layers
model.add(Dense(800, input_dim=784, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

# save model into file
# model_json = model.to_json()
# with open('recognition_handwritten_numbers.json', 'w') as f:
#     f.write(model_json)

# compile model
model.compile(loss='categorical_crossentropy', optimizer="SGD", metrics=["accuracy"])
print(model.summary())

# network training
# validation_split - 20% of data is test sample
model.fit(X_train, y_train, batch_size=200, nb_epoch=100, validation_split=0.2, verbose=1)

# save weights
# model.save_weights('recognition_handwritten_numbers.h5')


# transformation output from category to class label(digits from 0 to 9)
predictions = model.predict(X_train)
# predictions = np_utils.categorical_probas_to_classes(predictions)
