import tensorflow as tf
import numpy as np

tf.set_random_seed(777) # for reproducibility

# Predicting animal type based on various features
xy = np.loadtxt('D:/Bitcamp/Data/data-04-zoo.csv', delimiter = ',', dtype = np.float32)
x_data = xy[ : , 0 : -1]
y_data = xy[ : , [-1]]

nb_classes = 7 # 0 ~ 6

X = tf.placeholder(tf.loat32, [None, 16])
Y = tf.placeholder(tf.int32, [None, 1]) # 0 ~ 6

Y_one_hot = tf.one_hot(Y, nb_classes) # one hot
print("one_hot : ", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])
print("reshape one_hot : ", Y_one_hot)

'''
one_hot : Tensor("one_hot : 0", shape = (?, 1, 7), dtype = float32)
reshape one_hot : Tensor("Reshape : 0", shape = (?, 7), dtype = float32)
'''

W = tf.Variable(tf.random_normal([16, nb_classes]), name = 'weight')
b = tf.Variable(tf.random_normal([nb_classes]), name = 'bias')

# tf.nn.softmax computes softmax activations
# softmax = exp(logits) / reduce_sum(exp(logits), dim)
logits = tf.matmul(X * W) + b
hypothesis = tf.nn.softmax(logits)

# Cross entropy cost / loss fuction
