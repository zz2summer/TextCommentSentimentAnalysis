# -*- coding:utf-8 -*-
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[2.], input2:[4.]}))

# placeholder 和 feed_dict 绑定同时使用
