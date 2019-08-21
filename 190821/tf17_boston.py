import matplotlib.pyplot as plt
import numpy
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow

tensorflow.set_random_seed(777)

x = numpy.load("./Data/boston_housing_x.npy")
y = numpy.load("./Data/boston_housing_y.npy")

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 66)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)