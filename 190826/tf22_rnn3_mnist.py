import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("./mnist/data", one_hot = True)

# 옵션 설정
learning_rate = 0.001
total_epoch = 30
batch_size = 128

# 가로 픽셀수를 n_input으로, 세로 픽셀수를 입력 단계인 n_step으로 설정하였습니다.
n_input = 28
n_step = 28
n_hidden = 128
n_class = 10

# 신경망 모델 구성
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# print(X) # (?, 28, 28)
