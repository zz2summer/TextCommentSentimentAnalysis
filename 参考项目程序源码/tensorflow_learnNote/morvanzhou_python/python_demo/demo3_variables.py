import tensorflow as tf

state = tf.Variable(0, name='count')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

# must init if define variables
init = tf.initialize_all_variables()

with tf.Session() as sess:
    # define variables must run init
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))