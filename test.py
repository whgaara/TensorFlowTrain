# coding : utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('source', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


def gen_W(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))


def gen_b(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def pooling(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_image = tf.reshape(x, [-1, 28, 28, 1])

# 卷积层1
conv1_W = gen_W([5, 5, 1, 32])
conv1_b = gen_b([32])
conv1 = tf.nn.relu(conv2d(x_image, conv1_W) + conv1_b)
conv1_pool = pooling(conv1)

# 卷积层2
conv2_W = gen_W([5, 5, 32, 64])
conv2_b = gen_b([64])
conv2 = tf.nn.relu(conv2d(conv1_pool, conv2_W) + conv2_b)
conv2_pool = pooling(conv2)

new_x = tf.reshape(conv2_pool, [-1, 7*7*64])
# 全连接1
W1 = gen_W([7*7*64, 1024])
b1 = gen_b([1024])
fc1 = tf.nn.relu(tf.matmul(new_x, W1) + b1)
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# 全连接2
W2 = gen_W([1024, 10])
b2 = gen_b([10])
prediction = tf.matmul(fc1_drop, W2) + b2

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

correct_list = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_list, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.7})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
        print(acc)
