# -*- coding:utf-8 -*-
import tensorflow as tf
from python_demo.mnist import input_data

# 与之前一样，读入MNIST数据
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)


def add_layer(input, in_size, out_size, activation_Function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases

    if activation_Function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_Function(Wx_plus_b)

    return outputs


def computer_accurary(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accurary = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accurary, feed_dict={xs: v_xs, ys: v_ys})
    return result


# 定义数据
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# 添加输出层
prediction = add_layer(xs, 784, 10, activation_Function=tf.nn.softmax)

# 计算损失
# 下面会根据y和y_构造交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction)))
# 有了损失，就可以用梯度下降法针对模型的参数（W和b）进行优化
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 10 == 0:
        print(computer_accurary(mnist.test.images, mnist.test.labels))
