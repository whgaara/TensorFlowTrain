import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 使用numpy生成200个[-0.5, 0.5]间均匀分布的点
# 生成的点是一个一维数据，np.newaxis可以为其增加一个维度，其本质上就是一个none，详情请查阅numpy学习手册
x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
# np.random.normal是一个正态分布函数，参数1表示此概率分布的均值，参数2表示此概率分布的标准差，参数3是输出结果的形状
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise

# None表示是任意的形状
x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

'''
构建神经网络结构
'''
# 构建神经网络中间层，中间层有10个神经元
weights_l1 = tf.Variable(tf.random_normal([1, 10]))
blases_l1 = tf.Variable(tf.zeros([1, 10]))
Wx_plus_b_l1 = tf.matmul(x, weights_l1) + blases_l1
# tanh为激活函数：双曲正切函数
L1 = tf.nn.tanh(Wx_plus_b_l1)

# 定义输出层，输出层有1个神经元
weights_l2 = tf.Variable(tf.random_normal([10, 1]))
blases_l2 = tf.Variable(tf.zeros([1, 1]))
Wx_plus_b_l2 = tf.matmul(L1, weights_l2) + blases_l2
prediction = tf.nn.tanh(Wx_plus_b_l2)

loss = tf.reduce_mean(tf.square(y - prediction))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step, feed_dict={x: x_data, y: y_data})
    prediction_value = sess.run(prediction, feed_dict={x: x_data})

    # 通过图形看预测结果
    plt.figure()
    plt.scatter(x_data, y_data)
    plt.plot(x_data, prediction_value, 'r-', lw=5)
    plt.show()
