import tensorflow as tf
import matplotlib.pyplot as plt
import random

tf.set_random_seed(777) # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# print(mnist.train.images)
# print(mnist.test.labels)
# print(mnist.train.images.shape)
# print(mnist.test.labels.shape)
# print(type(mnist.train.images))

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.get_variable("W1", shape = [784, 512], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
l1 = tf.sigmoid(tf.matmul(X, W1) + b1)
# l1 = tf.nn.softmax(tf.matmul(X, W1) + b1)
# l1 = tf.nn.relu(tf.matmul(X, W1) + b1)
d1 = tf.nn.dropout(l1, keep_prob = 1e-10)

W2 = tf.get_variable("W2", shape = [512, 256], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]))
l2 = tf.sigmoid(tf.matmul(l1, W2) + b2)
# l2 = tf.nn.softmax(tf.matmul(l1, W2) + b2)
# l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
d2 = tf.nn.dropout(l2, keep_prob = 1e-10)

W3 = tf.get_variable("W3", shape = [256, 128], initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([128]))
l3 = tf.sigmoid(tf.matmul(l2, W3) + b3)
# l3 = tf.nn.softmax(tf.matmul(l2, W3) + b3)
# l3 = tf.nn.relu(tf.matmul(l2, W3) + b3)
d3 = tf.nn.dropout(l3, keep_prob = 1e-10)

W4 = tf.get_variable("W4", shape = [128, 64], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([64]))
l4 = tf.sigmoid(tf.matmul(l3, W4) + b4)
# l4 = tf.nn.softmax(tf.matmul(l3, W4) + b4)
# l4 = tf.nn.relu(tf.matmul(l3, W4) + b4)
d4 = tf.nn.dropout(l4, keep_prob = 1e-10)

W5 = tf.get_variable("W5", shape = [64, 32], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([32]))
l5 = tf.sigmoid(tf.matmul(l4, W5) + b5)
# l5 = tf.nn.softmax(tf.matmul(l4, W5) + b5)
# l5 = tf.nn.relu(tf.matmul(l4, W5) + b5)
d5 = tf.nn.dropout(l5, keep_prob = 1e-10)

W6 = tf.get_variable("W6", shape = [32, 16], initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.random_normal([16]))
l6 = tf.sigmoid(tf.matmul(l5, W6) + b6)
# l6 = tf.nn.softmax(tf.matmul(l5, W6) + b6)
# l6 = tf.nn.relu(tf.matmul(l5, W6) + b6)
d6 = tf.nn.dropout(l6, keep_prob = 1e-10)

W7 = tf.get_variable("W7", shape = [16, 16], initializer = tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([16]))
l7 = tf.sigmoid(tf.matmul(l6, W7) + b7)
# l7 = tf.nn.softmax(tf.matmul(l6, W7) + b7)
# l7 = tf.nn.relu(tf.matmul(l6, W7) + b7)
d7 = tf.nn.dropout(l7, keep_prob = 1e-10)

W8 = tf.get_variable("W8", shape = [16, 16], initializer = tf.contrib.layers.xavier_initializer())
b8 = tf.Variable(tf.random_normal([16]))
l8 = tf.sigmoid(tf.matmul(l7, W8) + b8)
# l8 = tf.nn.softmax(tf.matmul(l7, W8) + b8)
# l8 = tf.nn.relu(tf.matmul(l7, W8) + b8)
d8 = tf.nn.dropout(l8, keep_prob = 1e-10)

W9 = tf.get_variable("W9", shape = [16, 16], initializer = tf.contrib.layers.xavier_initializer())
b9 = tf.Variable(tf.random_normal([16]))
l9 = tf.sigmoid(tf.matmul(l8, W9) + b9)
# l9 = tf.nn.softmax(tf.matmul(l8, W9) + b9)
# l9 = tf.nn.relu(tf.matmul(l8, W9) + b9)
d9 = tf.nn.dropout(l9, keep_prob = 1e-10)

W10 = tf.get_variable("W10", shape = [16, 10], initializer = tf.contrib.layers.xavier_initializer())
b10 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.nn.softmax(tf.matmul(l9, W10) + b10)

# W1 = tf.get_variable("W1", shape = [?, ?], initializer = tf.random_uniform_initializer())
# b1 = tf.Variable(tf.random_normal([512]))
# L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
# L1 - tf.nn.dropout(L1, keep_prob = keep_prob)

tf.constant_initializer()
tf.zeros_initializer()
tf.random_uniform_initializer()
tf.random_normal_initializer()
tf.contrib.layers.xavier_initializer()

# Cross entropy cost / loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))

# train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

# Test model
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

# Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
num_epochs = 15
batch_size = 100
num_iterations = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations): # 550
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) # 100
            _, cost_val = sess.run([train, cost], feed_dict = {X : batch_xs, Y : batch_ys})
            avg_cost += cost_val / num_iterations
        
        print("Epoch : {:04d}, Cost : {:.9f}".format(epoch + 1, avg_cost))
    
    print("Learning finished")

    # Test the model using test sets
    print("Accuracy : ", accuracy.eval(session = sess, feed_dict = {X : mnist.test.images, Y : mnist.test.labels}))

    # Got one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label : ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
    print("Prediction : ", sess.run(tf.argmax(hypothesis, 1), feed_dict = {X : mnist.test.images[r : r + 1]}), )

    plt.imshow(mnist.test.images[r : r + 1].reshape(28, 28), cmap = "Greys", interpolation = "nearest")
    plt.show()