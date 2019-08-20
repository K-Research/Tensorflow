import numpy
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow

def RMAE(y_data, predict):
    return numpy.sqrt(mean_absolute_error(y_data, predict))

def RMSE(y_data, predict):
    return numpy.sqrt(mean_squared_error(y_data, predict))

xy = numpy.array([[828.659973, 833.450012, 908100, 828.349976, 831.659973], 
                [823.02002, 828.070007, 1828100, 821.655029, 828.070007], 
                [819.929993, 824.400024, 1438100, 818.97998, 824.159973], 
                [816, 820.958984, 1008100, 815.48999, 819.23999], 
                [819.359985, 823, 1188100, 818.469971, 818.97998], 
                [819, 823, 1198100, 816, 820.450012], 
                [811.700012, 815.25, 1098100, 809.780029, 813.669983], 
                [809.51001, 816.659973, 1398100, 804.539978, 809.559998]])

x_data = xy[ : ,  : -1]
y_data = xy[ : , [-1]]

# print(x_data.shape)
# print(y_data.shape)

X = tensorflow.placeholder("float", [None, 4])
Y = tensorflow.placeholder("float", [None, 1])

# W = tensorflow.Variable(tensorflow.random_normal([8, 4]))
# b = tensorflow.Variable(tensorflow.random_normal([1]))
W = tensorflow.Variable(tensorflow.random_normal([4, 1]))
b = tensorflow.Variable(tensorflow.random_normal([1]))

# hypothesis = W * X + b
hypothesis = tensorflow.matmul(X, W) + b

# cost / loss function
cost = tensorflow.reduce_mean(tensorflow.square(hypothesis - Y), axis = 1)

# optimizer
# train = tensorflow.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
train = tensorflow.train.AdamOptimizer(learning_rate = 0.1).minimize(cost)

# Launch graph
sess = tensorflow.Session()

# Initialize Tensorflow variables
sess.run(tensorflow.global_variables_initializer())

for step in range(10001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict = {X : x_data, Y : y_data})
    print(step, "Cost : ", cost_val, "\nPrediction : \n", hy_val)

predict = sess.run([hypothesis], feed_dict = {X : x_data})

predict = numpy.array(predict)
y_data_reshape = y_data.reshape((-1, ))
predict = predict.reshape((-1, ))

print("RMAE:", RMAE(y_data_reshape, predict))
print("RMSE:", RMSE(y_data_reshape, predict))
print("R2:", r2_score(y_data_reshape, predict))