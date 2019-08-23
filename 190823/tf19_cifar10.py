from keras.datasets import cifar10
from keras.utils import np_utils
import numpy
import random
import tensorflow as tf
from keras.datasets.cifar10 import load_data

# 다음 배치를 읽어오기 위한 next_batch 유틸리티 함수를 정의합니다.
def next_batch(number, data, labels):
  '''
  `num` 개수 만큼의 랜덤한 샘플들과 레이블들을 리턴합니다.
  '''
  index = numpy.arange(0 , len(data))
  numpy.random.shuffle(index)
  index = index[ : number]
  data_shuffle = [data[i] for i in index]
  labels_shuffle = [labels[i] for i in index]
  
  return numpy.asarray(data_shuffle), numpy.asarray(labels_shuffle)

tf.set_random_seed(777) # reproducibility

(x_train, y_train), (x_test, y_test) = load_data()

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

NB_CLASSES = 10
cifar10_train_num_examples = 50000

# print(x_train.shape) # (50000, 32, 32, 3)
# print(y_train.shape) # (50000, 1)
# print(x_test.shape) # (10000, 32, 32, 3)
# print(y_test.shape) # (10000, 1)

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
# y_train = np_utils.to_categorical(y_train, NB_CLASSES)
# y_test = np_utils.to_categorical(y_test, NB_CLASSES)

# y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10), axis = 1) # (50000, 1)
# y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10), axis = 1) # (10000, 1)

# input place holders
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.int32, [None, 1])

Y_one_hot = tf.one_hot(Y, NB_CLASSES) # one-hot
# print("one_hot:", Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NB_CLASSES])
# print("reshape one_hot:", Y_one_hot)

# L1 ImgIn shape = (?, 32, 32, 3)
W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev = 0.01))
L1 = tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding = 'SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# L2 ImgIn shape = (?, 16, 16, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
L2 = tf.nn.conv2d(L1, W2, strides = [1, 1, 1, 1], padding = 'SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

# L3 ImgIn shape = (?, 8, 8, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
L3 = tf.nn.conv2d(L2, W3, strides = [1, 1, 1, 1], padding = 'SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
L3_flat = tf.reshape(L3, [-1, 4 * 4 * 128])

W4 = tf.get_variable("W4", shape = [4 * 4 * 128, 10], initializer = tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L3_flat, W4) + b1

# define cost / loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model 
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = x_train.shape[0] // batch_size

    for i in range(total_batch):
            # batch_xs, batch_ys = next_batch(batch_size, x_train, y_train_one_hot.eval(session = sess))
            batch_xs, batch_ys = next_batch(batch_size, x_train, y_train)
            feed_dict = {X : batch_xs, Y : batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
            avg_cost += c / total_batch

    print('Epoch : ', '%04d' % (epoch + 1), 'cost = ', '{:.9f}'.format(avg_cost))

print('Learning Finished')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy : ', sess.run(accuracy, feed_dict = {X : x_test, Y : y_test}))

# Get one and predict
r = random.randint(0, 10000 - 1)
print("Label : ", sess.run(tf.argmax(x_test[r : r + 1], 1)))
print("Prediction : ", sess.run(tf.argmax(logits, 1), feed_dict = {X : x_test[r : r + 1]}))