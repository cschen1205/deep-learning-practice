import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.datasets import mnist
from keras.optimizers import Adam

np.random.seed(1671)

BATCH_SIZE = 128
EPOCHS = 20
NB_CLASSES = 10
RESHAPED = 784
OPTIMIZER = Adam()
VALIDATION_SPLIT = 0.2

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, RESHAPED)
x_test = x_test.reshape(10000, RESHAPED)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation('softmax'))

model.compile(optimizer=OPTIMIZER, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1, epochs=EPOCHS, validation_split=VALIDATION_SPLIT)

score = model.evaluate(x_test, y_test)

print('Test score: ', score[0])
print('Test accuracy: ', score[1])

