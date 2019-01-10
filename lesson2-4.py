import tensorflow as tf
import numpy as np


# 使用numpy生成100个随机点
# rand()无参时表示生成0-1间一个随机数，100表示生成100个0-1的随机数
# rand(n, m)则表示生成n*m的二维矩阵，依次类推
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

# 构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)
y = k * x_data + b

# 目的通过训练不断优化k和b，使得k和b不断接近0.1和0.2
# 二次代价函数
# y_data是训练样本值，y是预测值square是平方，reduce_mean是计算均值的函数，其详细原理建议自行了解
loss = tf.reduce_mean(tf.square(y_data - y))

# 定义一个梯度下降法优化器,0.2指的是学习率
'''
学习率越小，我们沿着损失梯度下降的速度越慢。从长远来看，这种谨慎慢行的选择可能还不错，
因为可以避免错过任何局部最优解，但它也意味着我们要花更多时间来收敛，尤其是如果我们处
于曲线的至高点。
'''
optimizer = tf.train.GradientDescentOptimizer(0.2)

# 定义一个最小化代价函数，这是训练的目的
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run([k, b]))
