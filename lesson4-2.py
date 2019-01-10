# coding: utf-8
# dropout
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
keep_prod = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 2000], stddev=0.1))
b1 = tf.Variable(tf.zeros([2000]) + 0.1)
# 双曲正切
L1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
# 设置工作神经元的百分比
L1_drop = tf.nn.dropout(L1, keep_prod)

W2 = tf.Variable(tf.truncated_normal([2000, 2000], stddev=0.1))
b2 = tf.Variable(tf.zeros([2000]) + 0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop, W2) + b2)
L2_drop = tf.nn.dropout(L2, keep_prod)

W3 = tf.Variable(tf.truncated_normal([2000, 1000], stddev=0.1))
b3 = tf.Variable(tf.zeros([1000]) + 0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop, W3) + b3)
L3_drop = tf.nn.dropout(L3, keep_prod)

W4 = tf.Variable(tf.truncated_normal([1000, 10], stddev=0.1))
b4 = tf.Variable(tf.zeros([10]) + 0.1)
L4 = tf.nn.tanh(tf.matmul(L3_drop, W4) + b4)
prediction = tf.nn.dropout(L4, keep_prod)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct_list = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_list, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(30):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prod: 0.7})
        test_acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prod: 1.0})
        train_acc = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels, keep_prod: 1.0})
        print(test_acc, train_acc)
