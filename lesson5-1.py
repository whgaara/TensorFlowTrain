# coding: utf-8
# 交叉熵
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 载入数据集，会自动下载到指定路径
mnist = input_data.read_data_sets('source', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次，//表示整除
n_batch = mnist.train.num_examples // batch_size

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
learn_rate = tf.Variable(1e-3, dtype=tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 500], stddev=0.1))
b1 = tf.Variable(tf.zeros([500]))
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
L1_drop = tf.nn.dropout(L1, 1.0)

W2 = tf.Variable(tf.truncated_normal([500, 300], stddev=0.1))
b2 = tf.Variable(tf.zeros([300]))
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, 1.0)

W3 = tf.Variable(tf.truncated_normal([300, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))
prediction = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

correct_list = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_list, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(51):
        sess.run(tf.assign(learn_rate, 0.001 * (0.95 ** epoch)))
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(acc, sess.run(learn_rate))
