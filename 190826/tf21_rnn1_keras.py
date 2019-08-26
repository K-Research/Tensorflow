from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import np_utils
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# 데이터 구축
idx2char = ['e', 'h', 'i', 'l', 'o']

_data = np.array([['h', 'i', 'h', 'e', 'l', 'l', 'o']], dtype = np.str).reshape(-1, 1)

# print(_data.shape # (7, 1))
# print(_data)
# print(_data.dtype)

enc = OneHotEncoder()
enc.fit(_data)
_data = enc.transform(_data).toarray().astype('float32') # scikit-learn의 OneHotEncoder를 실행하면 알파벳 순으로 함, float64에서 float32로 형 변환.

# print(_data)
# print(_data.shape) # (7, 5)
# print(type(_data))
# print(_data.type)

x_data = _data[ : 6, ] # (6, 5)
y_data = _data[1 : , ] # (6, 5)

# print(x_data)
# print(y_data)

x_data = x_data.reshape(1, 6, 5) # (1, 6, 5)
y_data = y_data.reshape(1, 6, 5)

# print(x_data.shape) # (1, 6, 5)
# print(x_data.dtype)
# print(y_data.shape) # (6, )

# 데이터 구성
# x : (batch_size, sequence_length, input_dim) 1, 6, 5
# 첫번째 아웃풋 : hidden_size = 2
# 첫번째 결과 : 1, 6, 5

num_classes = 5
batch_size = 1 # 전체행
sequence_length = 6 # 컬럼
input_dim = 5 # 몇개씩 작업
hidden_size = 5 #첫번째 노드 출력 갯수
learning_rate = 0.1

# 2. 모델 구성

model = Sequential()
model.add(LSTM(30, input_shape = (sequence_length, input_dim), return_sequences = True))
model.add(LSTM(10, return_sequences = True))
model.add(LSTM(input_dim, activation = 'softmax', return_sequences = True))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

early_stopping = EarlyStopping(monitor = 'loss', patience = 30, mode = 'auto')

model.fit(x_data, y_data, epochs = 600, batch_size = 1, callbacks = [early_stopping])

loss, accuracy = model.evaluate(x_data, y_data)
print('Accuracy : %.4f' %accuracy)

prediction = model.predict(x_data)
prediction = np.argmax(prediction, axis = 2)
print('Prediction : ', prediction)

result_str = [idx2char[c] for c in np.squeeze(prediction)]
print("\nPrediction str : ", ''.join(result_str))

# Accuracy : 1.0000
# Prediction :  [[2 1 0 3 3 4]]

# Prediction str :  ihello