import tensorflow as tf

tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

# cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train)) # loss = 'mse'

# optimizer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost) # optimizer = 'GD'

# Launch the graph in a session.
with tf.Session() as sess:
    # Initializes global variables in the graph.
    sess.run(tf.global_variables_initializer()) # tensorflow는 모델의 다른 연산을 실행하기 전에 반드시 명시적으로 초기화를 해야한다.

    # Fit the line
    for step in range(2001): # epoch = 2000
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)