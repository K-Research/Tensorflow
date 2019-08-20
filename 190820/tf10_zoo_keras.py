from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import numpy
import pandas
from sklearn.model_selection import train_test_split
import tensorflow

data_zoo = numpy.loadtxt('D:/Bitcamp/Data/data-04-zoo.csv', delimiter = ',', dtype = numpy.float32)

# print(data_zoo)

x_data = data_zoo[ : , 0 : -1]
y_data = data_zoo[ : , [-1]]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state = 66)

y_train = to_categorical(y_train, num_classes = 7, dtype = 'float32')
y_test = to_categorical(y_test, num_classes = 7, dtype = 'float32')

print(x_data.shape)
print(y_data.shape)

model = Sequential()
model.add(Dense(128, activation = 'relu', input_shape = (16, )))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 1, epochs = 100)

score = model.evaluate(x_test, y_test, batch_size = 1)
print("\nTest scroe : ", score[0])
print('Test accuracy : ', score[1])