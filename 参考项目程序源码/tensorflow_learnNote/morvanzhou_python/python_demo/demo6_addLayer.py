import tensorflow as tf

def add_layer(input, in_size, out_size, activation_Function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([out_size, 1]) + 0.1)
    Wx_plus_b = tf.matmul(input, Weights) + biases

    if activation_Function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_Function(Wx_plus_b)

    return outputs