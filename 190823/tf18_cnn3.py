import tensorflow as tf
import random
# import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777) # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot = True)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1]) # img 28 x 28 x 1 (black / white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape = (?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01)) # kernel_size = (3, 3), channel = 1, output = 32
print('W1 : ', W1)
# Conv -> (?, 28, 28, 32)
# Pool -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides = [1, 1, 1, 1], padding = 'SAME')
print('L1 : ', L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
print('L1 : ', L1)

# L2 ImgIn shape = (?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
# Conv -> (?, 14, 14, 64)
# Pool -> (?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# L3 ImgIn shape = (?, 4, 4, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# L4 ImgIn shape = (?, 2, 2, 128)
W4 = tf.Variable(tf.random_normal([3, 3, 128, 256], stddev = 0.01))
L4 = tf.nn.conv2d(L3, W4, strides = [1, 1, 1, 1], padding = 'SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L4_flat = tf.reshape(L4, [-1, 2 * 2 * 256])

# Final FC 2 x 2 x 256 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape = [2 * 2 * 256, 128], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([128]))
L5 = tf.matmul(L4_flat, W5) + b1

W6 = tf.get_variable("W6", shape = [128, 10], initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L5, W6) + b2

# define cost / loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X : batch_xs, Y : batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c / total_batch

    print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict = {X : mnist.test.images, Y : mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label : ", sess.run(tf.argmax(mnist.test.labels[r : r + 1], 1)))
print("Prediction : ", sess.run(tf.argmax(logits, 1), feed_dict = {X : mnist.test.images[r : r + 1]}))

# Accuracy :  0.9926