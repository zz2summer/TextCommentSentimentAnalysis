# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

def add_layer(input, in_size, out_size, activation_Function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('W'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('b'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(input, Weights) + biases

        if activation_Function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_Function(Wx_plus_b)

        return outputs


# 定义数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

layer1 = add_layer(xs, 1, 10, activation_Function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_Function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter('logs/', sess.graph)
sess.run(tf.initialize_all_variables())

print('over')

# tensorboard --logdir='logs/'
# H:/Codes/PyCharm_Projects/tensorflow/morvanzhou_python/python_demo/logs