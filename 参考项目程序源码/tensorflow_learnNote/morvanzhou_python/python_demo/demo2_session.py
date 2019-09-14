import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])

result = tf.matmul(matrix1, matrix2)

# method 1
'''
sess = tf.Session()
print(sess.run(result))
sess.close()
'''

# method 2
with tf.Session() as sess2:
    print(sess2.run(result))