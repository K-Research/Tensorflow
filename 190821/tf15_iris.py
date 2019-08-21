import matplotlib.pyplot as plt
import numpy
import random
from sklearn.model_selection import train_test_split
import tensorflow

tensorflow.set_random_seed(777)

dataset = numpy.load("./Data/iris2_data.npy")

x = dataset[ : , 0 : -1]
y = dataset[ : , [-1]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 66)

# print(x_train.shape)
# print(x_test.shape)
# print(y_train.shape)
# print(y_test.shape)

X = tensorflow.placeholder(tensorflow.float32, [None, 4])
Y = tensorflow.placeholder(tensorflow.float32, [None, 1])

W1 = tensorflow.get_variable(name = 'Weight1', shape = [4, 8], initializer = tensorflow.contrib.layers.xavier_initializer())
b1 = tensorflow.Variable(tensorflow.random_normal([8]), name = 'Bias1')
# b1 = tensorflow.Variable(tensorflow.zeros([8]), name = 'Bias1')
# l1 = tensorflow.nn.leaky_relu(tensorflow.matmul(X, W1) + b1)
l1 = tensorflow.sigmoid(tensorflow.matmul(X, W1) + b1)
# l1 = tensorflow.nn.softmax(tensorflow.matmul(X, W1) + b1)

W2 = tensorflow.get_variable(name = 'Weight2', shape = [8, 16], initializer = tensorflow.contrib.layers.xavier_initializer())
b2 = tensorflow.Variable(tensorflow.random_normal([16]), name = 'Bias2')
# b2 = tensorflow.Variable(tensorflow.zeros([16]), name = 'Bias2')
l2 = tensorflow.sigmoid(tensorflow.matmul(l1, W2) + b2)

W3 = tensorflow.get_variable(name = 'Weight3', shape = [16, 32], initializer = tensorflow.contrib.layers.xavier_initializer())
b3 = tensorflow.Variable(tensorflow.random_normal([32]), name = 'Bias3')
# b3 = tensorflow.Variable(tensorflow.zeros([32]), name = 'Bias3')
l3 = tensorflow.sigmoid(tensorflow.matmul(l2, W3) + b3)

W4 = tensorflow.get_variable(name = 'Weight4', shape = [32, 64], initializer = tensorflow.contrib.layers.xavier_initializer())
b4 = tensorflow.Variable(tensorflow.random_normal([64]), name = 'Bias4')
# b4 = tensorflow.Variable(tensorflow.zeros([64]), name = 'Bias4')
l4 = tensorflow.sigmoid(tensorflow.matmul(l3, W4) + b4)

W5 = tensorflow.get_variable(name = 'Weight5', shape = [64, 128], initializer = tensorflow.contrib.layers.xavier_initializer())
b5 = tensorflow.Variable(tensorflow.random_normal([128]), name = 'Bias5')
# b5 = tensorflow.Variable(tensorflow.zeros([128]), name = 'Bias5')
l5 = tensorflow.sigmoid(tensorflow.matmul(l4, W5) + b5)

W6 = tensorflow.get_variable(name = 'Weight6', shape = [128, 64], initializer = tensorflow.contrib.layers.xavier_initializer())
b6 = tensorflow.Variable(tensorflow.random_normal([64]), name = 'Bias6')
# b6 = tensorflow.Variable(tensorflow.zeros([64]), name = 'Bias6')
l6 = tensorflow.sigmoid(tensorflow.matmul(l5, W6) + b6)

W7 = tensorflow.get_variable(name = 'Weight7', shape = [64, 32], initializer = tensorflow.contrib.layers.xavier_initializer())
b7 = tensorflow.Variable(tensorflow.random_normal([32]), name = 'Bias7')
# b7 = tensorflow.Variable(tensorflow.zeros([32]), name = 'Bias7')
l7 = tensorflow.sigmoid(tensorflow.matmul(l6, W7) + b7)

W8 = tensorflow.get_variable(name = 'Weight8', shape = [32, 16], initializer = tensorflow.contrib.layers.xavier_initializer())
b8 = tensorflow.Variable(tensorflow.random_normal([16]), name = 'Bias8')
# b8 = tensorflow.Variable(tensorflow.zeros([16]), name = 'Bias8')
l8 = tensorflow.sigmoid(tensorflow.matmul(l7, W8) + b8)

W9 = tensorflow.get_variable(name = 'Weight9', shape = [16, 8], initializer = tensorflow.contrib.layers.xavier_initializer())
b9 = tensorflow.Variable(tensorflow.random_normal([8]), name = 'Bias9')
# b9 = tensorflow.Variable(tensorflow.zeros([8]), name = 'Bias9')
l9 = tensorflow.sigmoid(tensorflow.matmul(l8, W9) + b9)

W10 = tensorflow.get_variable(name = 'Weight10', shape = [8, 3], initializer = tensorflow.contrib.layers.xavier_initializer())
b10 = tensorflow.Variable(tensorflow.random_normal([3]), name = 'Bias10')
# b10 = tensorflow.Variable(tensorflow.zeros([3]), name = 'Bias10')
hypothesis = tensorflow.nn.softmax(tensorflow.matmul(l9, W10) + b10)

cost = tensorflow.reduce_mean(-tensorflow.reduce_sum(Y * tensorflow.log(hypothesis), axis = 1))
train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
predicted = tensorflow.equal(tensorflow.argmax(hypothesis, 1), tensorflow.argmax(x_test, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(predicted, dtype = tensorflow.float32))

# Launch graph
with tensorflow.Session() as session:

    # Initialize TensorFlow variables
    session.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = session.run([cost, train], feed_dict = {X : x_train, Y : y_train})
        
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = session.run([hypothesis, predicted, accuracy], feed_dict = {X : x_test, Y : y_test})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)