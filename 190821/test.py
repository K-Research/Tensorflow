import tensorflow as tf
import numpy as np
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

xy = np.load("./data/boston_housing_keras.npy")

x_data = xy[0][0]
y_data = xy[0][1]

x_test = xy[1][0]
y_test = xy[1][1]

print(x_data.shape)
print(x_test.shape)

tf.set_random_seed(777)

y_data = y_data.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
# print(x_data.shape, y_data.shape)
# x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

x = tf.placeholder(tf.float32, shape=[None, 13])
y = tf.placeholder(tf.float32, shape=[None, 1])

w = tf.Variable(tf.random_normal([1]))
b = tf.Variable(tf.random_normal([1]))

hypothesis = x*w + b

cost = tf.reduce_mean(tf.square(tf.subtract(hypothesis, y)))

train = tf.train.GradientDescentOptimizer(learning_rate=0.00001).minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x: x_test, y: y_test})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c , "\nAccuracy: ", a)

from sklearn.metrics import r2_score
r2_y_predict = r2_score(c, h)
print("R2: ", r2_y_predict)