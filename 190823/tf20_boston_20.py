import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def RMAE(y_test, y_):
    return np.sqrt(mean_absolute_error(y_test, y_))

def RMSE(y_test, y_):
    return np.sqrt(mean_squared_error(y_test, y_))

# tf.set_random_seed(777)
tf.set_random_seed(666)

boston_housing_x = np.load("./Data/boston_housing_x.npy")
boston_housing_y = np.load("./Data/boston_housing_y.npy")

sta = StandardScaler()
sta.fit(boston_housing_x)
boston_housing_x = sta.transform(boston_housing_x)
x_train = boston_housing_x

y_train = boston_housing_y.reshape((-1,1))
# y_train = np.array(y_train,dtype=np.int32)

# print(x_train.shape, y_train.shape)

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2)

# print(x_train.shape, y_train.shape)

X = tf.placeholder(tf.float32,[None, 13])
Y = tf.placeholder(tf.float32,[None, 1])

l1 = tf.layers.dense(X, 100, activation = tf.nn.relu)
l2 = tf.layers.dense(l1, 200, activation = tf.nn.relu)
l3 = tf.layers.dense(l2, 100, activation = tf.nn.relu)
hypothesis = tf.layers.dense(l3, 1, activation = tf.nn.relu)

cost=tf.reduce_mean(tf.square(hypothesis - Y))

# train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
# train = tf.train.AdadeltaOptimizer(learning_rate=0.05,).minimize(cost)

feed_dict = {X : x_train, Y : y_train}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(8501):
        # _, cost_val, w_val, b_val = sess.run([train, cost, w, b], feed_dict = feed_dict)
        _, cost_val = sess.run([train, cost], feed_dict = feed_dict)

        if step % 500 == 0:
            print("Step : {:5}\tCost : {:f}".format(step, cost_val))
            
    #### model predict#################################################
    # print(sess.run(hypothesis, feed_dict = feed_dict))
    pred = sess.run(hypothesis,feed_dict = {X : x_test})

    pred = np.array(pred)
    y_test = y_test.reshape((-1, ))
    pred = pred.reshape((-1, ))

    print("RMSE : ",RMSE(y_test, pred))
    print("RMAE : ",RMAE(y_test, pred))
    print("R2 : ",r2_score(y_test, pred))