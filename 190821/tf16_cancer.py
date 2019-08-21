import matplotlib.pyplot as plt
import numpy
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow

tensorflow.set_random_seed(777)

dataset = numpy.load("./Data/cancer_data.npy")

# print(dataset.shape)

x = dataset[ : , 0 : -1]
y = dataset[ : , [-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

X = tensorflow.placeholder(tensorflow.float32, [None, 30])
Y = tensorflow.placeholder(tensorflow.float32, [None, 1])

W1 = tensorflow.get_variable(name = 'Weight1', shape = [30, 64], initializer = tensorflow.zeros_initializer())
b1 = tensorflow.Variable(tensorflow.random_normal([64]), name = 'Bias1')
l1 = tensorflow.nn.leaky_relu(tensorflow.matmul(X, W1) + b1)
# l1 = tensorflow.sigmoid(tensorflow.matmul(X, W1) + b1)

W2 = tensorflow.get_variable(name = 'Weight2', shape = [64, 128], initializer = tensorflow.zeros_initializer())
b2 = tensorflow.Variable(tensorflow.random_normal([128]), name = 'Bias2')
# l2 = tensorflow.nn.leaky_relu(tensorflow.matmul(l1, W2) + b2)
l2 = tensorflow.sigmoid(tensorflow.matmul(l1, W2) + b2)

W3 = tensorflow.get_variable(name = 'Weight3', shape = [128, 64], initializer = tensorflow.zeros_initializer())
b3 = tensorflow.Variable(tensorflow.random_normal([64]), name = 'Bias3')
# l3 = tensorflow.nn.leaky_relu(tensorflow.matmul(l2, W3) + b3)
l3 = tensorflow.sigmoid(tensorflow.matmul(l2, W3) + b3)

W4 = tensorflow.get_variable(name = 'Weight4', shape = [64, 32], initializer = tensorflow.zeros_initializer())
b4 = tensorflow.Variable(tensorflow.random_normal([32]), name = 'Bias4')
# l4 = tensorflow.nn.leaky_relu(tensorflow.matmul(l3, W4) + b4)
l4 = tensorflow.sigmoid(tensorflow.matmul(l3, W4) + b4)

W5 = tensorflow.get_variable(name = 'Weight5', shape = [32, 16], initializer = tensorflow.zeros_initializer())
b5 = tensorflow.Variable(tensorflow.random_normal([16]), name = 'Bias5')
# l5 = tensorflow.nn.leaky_relu(tensorflow.matmul(l4, W5) + b5)
l5 = tensorflow.sigmoid(tensorflow.matmul(l4, W5) + b5)

W6 = tensorflow.get_variable(name = 'Weight6', shape = [16, 8], initializer = tensorflow.zeros_initializer())
b6 = tensorflow.Variable(tensorflow.random_normal([8]), name = 'Bias6')
# l6 = tensorflow.nn.leaky_relu(tensorflow.matmul(l5, W6) + b6)
l6 = tensorflow.sigmoid(tensorflow.matmul(l5, W6) + b6)

W7 = tensorflow.get_variable(name = 'Weight7', shape = [8, 4], initializer = tensorflow.zeros_initializer())
b7 = tensorflow.Variable(tensorflow.random_normal([4]), name = 'Bias7')
# l7 = tensorflow.nn.leaky_relu(tensorflow.matmul(l6, W7) + b7)
l7 = tensorflow.sigmoid(tensorflow.matmul(l6, W7) + b7)

W8 = tensorflow.get_variable(name = 'Weight8', shape = [4, 2], initializer = tensorflow.zeros_initializer())
b8 = tensorflow.Variable(tensorflow.random_normal([2]), name = 'Bias8')
# l8 = tensorflow.nn.leaky_relu(tensorflow.matmul(l7, W8) + b8)
l8 = tensorflow.sigmoid(tensorflow.matmul(l7, W8) + b8)

W9 = tensorflow.get_variable(name = 'Weight9', shape = [2, 1], initializer = tensorflow.zeros_initializer())
b9 = tensorflow.Variable(tensorflow.random_normal([1]), name = 'Bias9')

# Hypothesis using sigmoid : tensorflow.div(1., 1. + tensorflow.exp(tensorflow.matmul(X, W)))
hypothesis = tensorflow.sigmoid(tensorflow.matmul(l8, W9) + b9) # activation = 'sigmoid'

# cost / loss fuction 로지스틱 리그레션에서 cost에 -가 붙는다.
cost = -tensorflow.reduce_mean(Y * tensorflow.log(hypothesis) + (1 - Y) * tensorflow.log(1 - hypothesis)) # loss = 'binary_crossentropy'

# train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.000008).minimize(cost)

predicted = tensorflow.cast(hypothesis > 0.5, dtype = tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype = tensorflow.float32))

# Launch graph
with tensorflow.Session() as sess:

    # Initialize TensorFlow variables
    sess.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict = {X : x_train, Y : y_train})
        
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_test, Y : y_test})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)