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

# W = tf.Variable(tf.random_normal([784, 10]))
# b = tf.Variable(tf.random_normal([10]))
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# Cross entropy cost / loss
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

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

# Accuracy :  0.9219