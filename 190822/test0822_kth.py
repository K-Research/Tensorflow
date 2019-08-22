import numpy
import pandas
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow

dataset = pandas.read_csv('D:/Bitcamp/Data/test0822.csv', usecols = ['kp_0h', 'kp_3h', 'kp_6h', 'kp_9h', 'kp_12h', 'kp_15h', 'kp_18h', 'kp_21h'])

# print(dataset)
# print(dataset.shape) # (5479, 8)

dataset = numpy.array(dataset)

previous_dataset = dataset[0 : 3113,  : ] # (3113, 8)
subsequent_dataset = dataset[3119 : ,  : ] # (2360, 8)

# print(previous_dataset)
# print(previous_dataset.shape)
# print(subsequent_dataset)
# print(subsequent_dataset.shape)

previous_dataset = previous_dataset.reshape(3113 * 8, 1) # (24904, 1)
subsequent_dataset = subsequent_dataset.reshape(2360 * 8, 1) # (18880, 1)

# print(previous_dataset.shape)
# print(subsequent_dataset.shape)

dataset_for_predict = previous_dataset # (24904, 1)

def split(seq, size):
    list = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i + size)]
        list.append(subset)
    return list

previous_dataset = split(previous_dataset, 5)
previous_dataset = numpy.array(previous_dataset) # (24900, 5, 1)

subsequent_dataset = split(subsequent_dataset, 5)
subsequent_dataset = numpy.array(subsequent_dataset) # (18876, 5, 1)

# print(previous_dataset.shape)
# print(subsequent_dataset.shape)

previous_dataset = previous_dataset.reshape(24900, 5 * 1)
subsequent_dataset = subsequent_dataset.reshape(18876, 5 * 1)

# print(previous_dataset.shape)
# print(subsequent_dataset.shape)

total_dataset = numpy.concatenate((previous_dataset, subsequent_dataset)) # (43776, 5)

# print(total_dataset.shape)

total_dataset = total_dataset.reshape(43776 * 5, 1)

# scaler = MinMaxScaler()
scaler = StandardScaler()

scaler.fit(total_dataset)
total_dataset = scaler.transform(total_dataset)

total_dataset = total_dataset.reshape(43776, 5)

train_dataset1 = total_dataset[ : 24860,  : ]
train_dataset2 = total_dataset[24900 : ,  : ]

train_dataset = numpy.concatenate((train_dataset1, train_dataset2)) # (43736, 5)

# print(train_dataset.shape)

dataset_for_predict = total_dataset[24860 : 24900,  : ] # (40, 5)

# print(dataset_for_predict.shape)

x_train = train_dataset[ : , 0 : 4] # (43736, 4)
y_train = train_dataset[ : , 4 : ] # (43736, 1)

# print(x_train.shape)
# print(y_train.shape)

x_for_predict = dataset_for_predict[ : , 0 : 4] # (40, 4)
y_for_predict = dataset_for_predict[ : , 4 : ] # (40, 1)

# print(x_for_predict.shape)
# print(y_for_predict.shape)

tensorflow.set_random_seed(777)

X = tensorflow.placeholder(tensorflow.float32, [None, 4])
Y = tensorflow.placeholder(tensorflow.float32, [None, 1])

def parametric_relu(_x, name):
    # with tensorflow.variable_scope("parametric_relu", reuse = tensorflow.AUTO_REUSE):
    alphas = tensorflow.get_variable(name, _x.get_shape()[-1], initializer = tensorflow.constant_initializer(0.0), dtype = tensorflow.float32)
    pos = tensorflow.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

count = 1

def layer(input, output, previous_layer, name, dropout = 0, end = False):
    
    global count

    W = tensorflow.get_variable("w%d"%(count), shape = [input, output], initializer = tensorflow.constant_initializer())
    b = tensorflow.Variable(tensorflow.random_normal([output]))

    layer = parametric_relu(tensorflow.matmul(previous_layer, W) + b, name)

    if dropout != 0:
        layer = tensorflow.nn.dropout(layer, keep_prob = dropout)

    count += 1

    return layer

layer1 = layer(4, 16, X, 'PReLU1')
layer2 = layer(16, 64, layer1, 'PReLU2')
layer3 = layer(64, 16, layer2, 'PReLU3')
layer4 = layer(16, 4, layer3, 'PReLU4')
hypothesis = layer(4, 1, layer4, 'PReLU5', end = True)

cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - Y))

train = tensorflow.train.AdamOptimizer(learning_rate = 1e-3).minimize(cost)

session = tensorflow.Session()

session.run(tensorflow.global_variables_initializer())

def RMAE(y_data, prediction):
    return numpy.sqrt(mean_absolute_error(y_for_predict, prediction))

def RMSE(y_data, prediction):
    return numpy.sqrt(mean_squared_error(y_for_predict, prediction))

for step in range(10001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train], feed_dict = {X : x_train, Y : y_train})
    print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)

prediction = session.run([hypothesis], feed_dict = {X : x_for_predict})

prediction = numpy.array(prediction) # (1, 40, 1)
prediction = prediction.reshape(40, 1)
prediction = scaler.inverse_transform(prediction)
prediction = numpy.rint(prediction)
y_for_predict_reshaped = y_for_predict.reshape((-1, ))
prediction = prediction.reshape((-1, ))

print("RMAE :", RMAE(y_for_predict_reshaped, prediction))
print("RMSE :", RMSE(y_for_predict_reshaped, prediction))
print("R2 :", r2_score(y_for_predict_reshaped, prediction))

def save_prediction ():
    prediction_reshaped = prediction.reshape(5, 8)

    kp_0h = prediction_reshaped[ : , [0]]
    kp_3h = prediction_reshaped[ : , [1]]
    kp_6h = prediction_reshaped[ : , [2]]
    kp_9h = prediction_reshaped[ : , [3]]
    kp_12h = prediction_reshaped[ : , [4]]
    kp_15h = prediction_reshaped[ : , [5]]
    kp_18h = prediction_reshaped[ : , [6]]
    kp_21h = prediction_reshaped[ : , [7]]

    kp_0h = numpy.ndarray.tolist(kp_0h)
    kp_3h = numpy.ndarray.tolist(kp_3h)
    kp_6h = numpy.ndarray.tolist(kp_6h)
    kp_9h = numpy.ndarray.tolist(kp_9h)
    kp_12h = numpy.ndarray.tolist(kp_12h)
    kp_15h = numpy.ndarray.tolist(kp_15h)
    kp_18h = numpy.ndarray.tolist(kp_18h)
    kp_21h = numpy.ndarray.tolist(kp_21h)

    numpy.savetxt('test0822_kth.csv', numpy.c_[kp_0h, kp_3h, kp_6h, kp_9h, kp_12h, kp_15h, kp_18h, kp_21h], fmt = '%i', 
                    delimiter = ',', header = 'kp_0h, kp_3h, kp_6h, kp_9h, kp_12h, kp_15h, kp_18h, kp_21h', comments = '')
    return

save_prediction()