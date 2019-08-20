import numpy
import pandas
import tensorflow

data_zoo = numpy.loadtxt('D:/Bitcamp/Data/data-04-zoo.csv', delimiter = ',', dtype = numpy.float32)

# print(data_zoo)

x_data = data_zoo[ : , 0 : -1]
y_data = data_zoo[ : , [-1]]

# print(x_data.shape)
# print(y_data.shape)

X = tensorflow.placeholder("float", [None, 16])
Y = tensorflow.placeholder("float", [None, 1])

W = tensorflow.Variable(tensorflow.random_normal([16, 7]), name = 'weight')
b = tensorflow.Variable(tensorflow.random_normal([7]), name = 'bias')

# Hypothesis using sigmoid : tensorflow.div(1., 1. + tensorflow.exp(tensorflow.matmul(X, W)))
hypothesis = tensorflow.sigmoid(tensorflow.matmul(X, W) + b) # activation = 'sigmoid'

# cost / loss fuction 로지스틱 리그레션에서 cost에 -가 붙는다.
cost = -tensorflow.reduce_mean(Y * tensorflow.log(hypothesis) + (1 - Y) * tensorflow.log(1 - hypothesis)) # loss = 'binary_crossentropy'

train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# Accuracy computation
# True if hypothesis > 0.5 else False
predicted = tensorflow.cast(hypothesis > 0.5, dtype = tensorflow.float32)
accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(predicted, Y), dtype = tensorflow.float32))

# Launch graph
with tensorflow.Session() as session:
    # Initialize TensorFlow variables
    session.run(tensorflow.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = session.run([cost, train], feed_dict = {X : x_data, Y : y_data})
        
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = session.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
    print("\nHypothesis : ", h, "\nCorrect (Y) : ", c, "\nAccuracy : ", a)