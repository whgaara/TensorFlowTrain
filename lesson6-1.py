import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('source', one_hot=True)
batch_size = 100
# tf.flags.DEFINE_integer('batch_size', 100, '')
# FLAGS = tf.flags.FLAGS
n_batch = mnist.train.num_examples // batch_size
keep_prob = tf.placeholder(tf.float32)


# 初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 初始化偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积层
def conv2d(x, W):
    # strides的第一项和第四项的值固定为1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# 池化层
def max_pool_2x2(x):
    # ksize表示窗口的大小
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, [None, 10])
# 将序列化的x转为2d的图形形状[batch, height, weight, channels]，因为x是placeholder所以-1表示使用默认值，该是多少就是多少
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 初始化第一个卷积层的权值和偏置
# 卷积核的尺寸为5*5，共有32个卷积核，1表示通道数，黑白图是1，彩色的是3（RGB）
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# 第一层卷积和池化
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 初始化第二个卷积层的权值和偏置
# 上层有32个卷积核，就会产生32个不同的平面，相当于32个通道
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = weight_variable([64])

# 第二层卷积池化
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
此时，一张28*28的图片经过卷积后，由于padding使用的是SAME，因此平面的大小不变，但是会产生从32到64个平面，
每个平面再经过池化，会由28*28变成14*14到7*7，所以最后的池化结果为：28*28*1变成7*7*64
'''

# 初始化第一个全连接权值和偏置
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# 扁平化最终池化输出结果, 如果有100张图-1处就是100
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

# 执行第一个全连接层
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 执行第二个全连接层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# =============================================================== #
# prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
prediction = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# 定义代价函数
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
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
