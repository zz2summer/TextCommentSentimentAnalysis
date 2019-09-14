# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# 创建数据集
x_data = np.random.rand(100).astype((np.float32))
# really y
y_data = x_data * 0.1 + 0.3

# test structure
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y_test = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y_test-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)
# test structure

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
