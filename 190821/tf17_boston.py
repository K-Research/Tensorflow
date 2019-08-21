import matplotlib.pyplot as plt
import numpy
import random
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow

def RMAE(y_data, predict):
    return numpy.sqrt(mean_absolute_error(y_test, predict))

def RMSE(y_data, predict):
    return numpy.sqrt(mean_squared_error(y_test, predict))

def parametric_relu(_x, name):
    # with tensorflow.variable_scope("parametric_relu", reuse = tensorflow.AUTO_REUSE):
    alphas = tensorflow.get_variable(name, _x.get_shape()[-1], initializer = tensorflow.constant_initializer(0.0), dtype = tensorflow.float32)
    pos = tensorflow.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5
    return pos + neg

tensorflow.set_random_seed(777)

x = numpy.load("./Data/boston_housing_x.npy")
y = numpy.load("./Data/boston_housing_y.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X = tensorflow.placeholder(tensorflow.float32, [None, 13])
Y = tensorflow.placeholder(tensorflow.float32, [None, 1])

W1 = tensorflow.get_variable(name = 'Weight1', shape = [13, 4], initializer = tensorflow.zeros_initializer())
b1 = tensorflow.Variable(tensorflow.random_normal([4]), name = 'Bias1')
# l1 = tensorflow.nn.leaky_relu(tensorflow.matmul(X, W1) + b1)
# l1 = tensorflow.keras.layers.PReLU(tensorflow.matmul(X, W1) + b1)
l1 = parametric_relu(tensorflow.matmul(X, W1) + b1, 'PReLU1')
# l1 = tensorflow.sigmoid(tensorflow.matmul(X, W1) + b1)

W2 = tensorflow.get_variable(name = 'Weight2', shape = [4, 2], initializer = tensorflow.zeros_initializer())
b2 = tensorflow.Variable(tensorflow.random_normal([2]), name = 'Bias2')
# l2 = tensorflow.nn.leaky_relu(tensorflow.matmul(l1, W2) + b2)
l2 = parametric_relu(tensorflow.matmul(l1, W2) + b2, 'PReLU2')
# l2 = tensorflow.sigmoid(tensorflow.matmul(l1, W2) + b2)

W3 = tensorflow.get_variable(name = 'Weight3', shape = [2, 1], initializer = tensorflow.zeros_initializer())
b3 = tensorflow.Variable(tensorflow.random_normal([1]), name = 'Bias3')
# hypothesis = tensorflow.nn.leaky_relu(tensorflow.matmul(l2, W3) + b3)
hypothesis = parametric_relu(tensorflow.matmul(l2, W3) + b3, 'PReLU3')

# Simplified cost / loss function
cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - Y))

# Minimize
# train = tensorflow.train.GradientDescentOptimizer(learning_rate = 1e-5).minimize(cost)
train = tensorflow.train.AdamOptimizer(learning_rate = 1e-2).minimize(cost)

# Launch the graph in a session.
session = tensorflow.Session()

# Initializes global variables in the graph.
session.run(tensorflow.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = session.run([cost, hypothesis, train], feed_dict = {X : x_train, Y : y_train})
    print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)

predict = session.run([hypothesis], feed_dict = {X : x_test})

predict = numpy.array(predict)
y_test_reshape = y_test.reshape((-1, ))
predict = predict.reshape((-1, ))

print("RMAE:", RMAE(y_test_reshape, predict))
print("RMSE:", RMSE(y_test_reshape, predict))
print("R2:", r2_score(y_test_reshape, predict))

# RMAE: 1.670594350652709
# RMSE: 4.298444520887089
# R2: 0.7936536607814402