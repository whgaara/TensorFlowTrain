# coding: utf-8
# LSTM

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def RNN(x, weights, biases):
    # inputs = [batch_size, max_time, n_inputs]
    inputs = tf.reshape(x, shape=[-1, max_time, n_inputs])
    # 定义基本的cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


mnist = input_data.read_data_sets('source', one_hot=True)
batch_size = 50
n_batch = mnist.train.num_examples // batch_size

n_inputs = 28
max_time = 28
n_classes = 10
lstm_size = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

# 这个变量必须在声明了Variable以后再定义
saver = tf.train.Saver()

prediction = RNN(x, weights, biases)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(6):
        for batch in range(n_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print(acc)
    # 将训练的模型保存下来，如果要使用训练的结果，可调用saver.restore
    saver.save(sess, 'net/my_net.ckpt')
