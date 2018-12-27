import input_data
import tensorflow as tf


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()
x = tf.placeholder("float", shape=[None, 784], name="x")
y_ = tf.placeholder("float", shape=[None, 10], name="labels")
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="W")


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="B")


def conv2d(x1, W1):
    return tf.nn.conv2d(x1, W1, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x2):
    return tf.nn.max_pool(x2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope(name="conv1"):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


with tf.name_scope(name="conv2"):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


with tf.name_scope(name="fc1"):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


with tf.name_scope(name="fc2"):
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


tf.summary.histogram("weights", W_conv1)
tf.summary.histogram("weights", W_conv2)
tf.summary.histogram("biases", b_conv2)
tf.summary.histogram("biases", b_conv1)
tf.summary.histogram("activation", h_conv1)
tf.summary.histogram("activation", h_conv2)

with tf.name_scope(name="cross_entropy"):
    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope(name="accuracy"):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
tf.summary.scalar('accuracy', accuracy)

for learning_rates in [1e-3, 1e-4, 1e-5]:
    with tf.name_scope(name="train"):
        train_step = tf.train.AdamOptimizer(learning_rates).minimize(cross_entropy)
    sess.run(tf.initialize_all_variables())
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter("/tmp/log/mnist")
    writer.add_graph(sess.graph)
    for i in range(500):
        batch = mnist.train.next_batch(50)
        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            writer.add_summary(s, i)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step %d, training accuracy %g" % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    writer.close()
    print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
