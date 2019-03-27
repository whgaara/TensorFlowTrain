import tensorflow as tf


# 创建两个常量Tensor
m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [3]])

# 创建一个矩阵乘法op
product = tf.matmul(m1, m2)
print(product)

# 定义一个会话,启动默认图
sess = tf.Session()

# 通过run方法来启动这个
result = sess.run(product)
print(result)

sess.close()
