import tensorflow as tf


# Build your graph.
x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
w = tf.Variable(tf.random_uniform([2, 2]))
y = tf.matmul(x, w)
y_ = tf.constant([[0, 0], [1, 1]])

loss = tf.losses.absolute_difference(y_, y)
train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    # C:\Users\cooma\AppData\Roaming\Python\Python36\Scripts
    writer = tf.summary.FileWriter("/logs", sess.graph)
    init = tf.global_variables_initializer()
    sess.run(init)
    for i in range(1000):
        sess.run(train_op)
    writer.close()
