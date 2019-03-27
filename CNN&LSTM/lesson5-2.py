# coding: utf-8
# tensor board

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# 载入数据集，会自动下载到指定路径
mnist = input_data.read_data_sets('source', one_hot=True)

# 每个批次的大小
batch_size = 100
# 计算一共有多少批次，//表示整除
n_batch = mnist.train.num_examples // batch_size

# 命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    y = tf.placeholder(tf.float32, [None, 10], name='y')

with tf.name_scope('layer'):
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    prediction = tf.nn.softmax(tf.matmul(x, W) + b)

# 二次代价函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))

# 梯度下降
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

correct_rate = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_rate, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for epoch in range(1):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print('当前的准确率为：%s' % acc)
