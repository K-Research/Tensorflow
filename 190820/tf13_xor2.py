import tensorflow as tf
import numpy as np

tf.set_random_seed(777) # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype = np.float32)

# X, Y, W, b, hypothesis, cost, train
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])

W1 = tf.Variable(tf.random_normal([2, 512]), name = "weight1")
b1 = tf.Variable(tf.random_normal([512]), name = "bias1")
layer1 = tf.sigmoid(tf.matmul(X, W1) + b1)

W2 = tf.Variable(tf.random_normal([512, 256]), name = "weight2")
b2 = tf.Variable(tf.random_normal([256]), name = "bias2")
layer2 = tf.sigmoid(tf.matmul(layer1, W2) + b2)

W3 = tf.Variable(tf.random_normal([256, 128]), name = "weight3")
b3 = tf.Variable(tf.random_normal([128]), name = "bias3")
layer3 = tf.sigmoid(tf.matmul(layer2, W3) + b3)

W4 = tf.Variable(tf.random_normal([128, 64]), name = "weight4")
b4 = tf.Variable(tf.random_normal([64]), name = "bias4")
layer4 = tf.sigmoid(tf.matmul(layer3, W4) + b4)

W5 = tf.Variable(tf.random_normal([64, 32]), name = "weight5")
b5 = tf.Variable(tf.random_normal([32]), name = "bias5")
layer5 = tf.sigmoid(tf.matmul(layer4, W5) + b5)

W6 = tf.Variable(tf.random_normal([32, 16]), name = "weight6")
b6 = tf.Variable(tf.random_normal([16]), name = "bias6")
layer6 = tf.sigmoid(tf.matmul(layer5, W6) + b6)

W7 = tf.Variable(tf.random_normal([16, 8]), name = "weight7")
b7 = tf.Variable(tf.random_normal([8]), name = "bias7")
layer7 = tf.sigmoid(tf.matmul(layer6, W7) + b7)

W8 = tf.Variable(tf.random_normal([8, 4]), name = "weight8")
b8 = tf.Variable(tf.random_normal([4]), name = "bias8")
layer8 = tf.sigmoid(tf.matmul(layer7, W8) + b8)

W9 = tf.Variable(tf.random_normal([4, 2]), name = "weight9")
b9 = tf.Variable(tf.random_normal([2]), name = "bias9")
layer9 = tf.sigmoid(tf.matmul(layer8, W9) + b9)

W10 = tf.Variable(tf.random_normal([2, 1]), name = "weight10")
b10 = tf.Variable(tf.random_normal([1]), name = "bias10")
hypothesis = tf.sigmoid(tf.matmul(layer9, W10) + b10)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        _, cost_val, W1_val, W2_val, W3_val, W4_val, W5_val, W6_val, W7_val, W8_val, W9_val, W10_val = sess.run([train, cost, W1, W2, W3, W4, W5, W6, W7, W8, W9, W10], 
                                                                                                                feed_dict = {X : x_data, Y : y_data})

        if step % 100 == 0:
            print(step, cost_val, W1_val, W2_val, W3_val, W4_val, W5_val, W6_val, W7_val, W8_val, W9_val, W10_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict = {X : x_data, Y : y_data})
    print("\nhypothesis : ", h, "\nCorrect : ", c, "\nAccuracy : ", a)