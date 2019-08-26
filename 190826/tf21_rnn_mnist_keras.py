from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils
import numpy

image_rows = 28
image_columns = 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# print(x_train.shape) # (60000, 28, 28)
# print(y_train.shape) # (60000, )
# print(x_test.shape) # (10000, 28, 28)
# print(y_test.shape) # (10000, )

model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape = (28, 28))
model.add(Dense(10, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')

model.fit(x_train, y_train, epochs = 10, batch_size = 28, callbacks = [early_stopping])

loss, accuracy = model.evaluate(x_test, y_test)
print('Accuracy : %.4f'%accuracy)

# Accuracy : 0.9453