from keras.callbacks import EarlyStopping
from keras.layers import Dense, LeakyReLU, LSTM
from keras.models import Sequential
import numpy
from sklearn.metrics import mean_squared_error, r2_score

def RMSE(y_test, y_predict):
    return numpy.sqrt(mean_squared_error(y_test, y_predict))

def split(seq, size):
    list = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        list.append(subset)
    return list

dataset = []
test_dataset = []

for i in range(1, 101):
    dataset.append(i)

for j in range(101, 111):
    test_dataset.append(j)

size = 7

dataset_split = split(dataset, size)
dataset_split = numpy.array(dataset_split) # (94, 7)

test_dataset_split = split(test_dataset, size)
test_dataset_split = numpy.array(test_dataset_split)

# print(dataset_split.shape)

x_train = dataset_split[ : ,  : 6] # (94, 6)
y_train = dataset_split[ : , [-1]] # (94, 1)

x_test = test_dataset_split[ : ,  : 6]
y_test = test_dataset_split[ : , [-1]]

# print(x_train.shape)
# print(y_train.shape)

x_train = x_train.reshape(94, 6, 1)
# y_train = y_train.reshape(94, 1, 1)
x_test = x_test.reshape(4, 6, 1)

model = Sequential()

model.add(LSTM(128, input_shape = (6, 1)))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(64))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(32))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(16))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(8))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(4))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(2))
model.add(LeakyReLU(alpha = 0.3))
model.add(Dense(1))
model.add(LeakyReLU(alpha = 0.3))

model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mse'])

early_stopping = EarlyStopping(monitor = 'mean_squared_error', patience = 30, mode = 'auto')

model.fit(x_train, y_train, epochs = 1000, batch_size = 1, callbacks = [early_stopping])

y_predict = model.predict(x_test)
r2_y_predict = r2_score(y_test, y_predict)

print("RMSE : ", RMSE(y_test, y_predict))
print("R2 : ", r2_y_predict)
print("Predict : ", y_predict)