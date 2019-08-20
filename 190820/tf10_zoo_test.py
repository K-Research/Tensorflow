import keras
import numpy
import pandas
import tensorflow

data_zoo = numpy.loadtxt('D:/Bitcamp/Data/data-04-zoo.csv', delimiter = ',', dtype = numpy.float32)

# print(data_zoo)

x_data = data_zoo[ : , 0 : -1]
y_data = data_zoo[ : , [-1]]

y_data = keras.utils.to_categorical(y_data, num_classes = 7, dtype = 'float32')

# print(x_data.shape)
# print(y_data.shape)

X = tensorflow.placeholder("float", [None, 16])
Y = tensorflow.placeholder("float", [None, 7])

W = tensorflow.Variable(tensorflow.random_normal([16, 7]), name = 'weight')
b = tensorflow.Variable(tensorflow.random_normal([7]), name = 'bias')

# tensorflow.nn.softmax computes fotmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
hypothesis = tensorflow.nn.softmax(tensorflow.matmul(X, W) + b) # activation = 'softmax'

# Cross entropy cost / loss
cost = tensorflow.reduce_mean(-tensorflow.reduce_sum(Y * tensorflow.log(hypothesis), axis = 1)) # loss = 'categorical_crossentropy'

train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

predicted = tensorflow.equal(tensorflow.argmax(hypothesis, 1), tensorflow.argmax(y_data, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(predicted, dtype = tensorflow.float32))

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