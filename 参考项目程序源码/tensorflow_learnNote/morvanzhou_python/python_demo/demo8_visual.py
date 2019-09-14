# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
# 导入可视化模块
import matplotlib.pyplot as plt

def add_layer(input, in_size, out_size, activation_Function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
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

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

layer1 = add_layer(xs, 1, 10, activation_Function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_Function=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 生成图片框
fig = plt.figure()
# 编号
ax = fig.add_subplot(1, 1, 1)
# 以点的形式展示
ax.scatter(x_data, y_data)
# 显示图片后继续执行程序不暂停
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        # to see every improement
        # print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)

