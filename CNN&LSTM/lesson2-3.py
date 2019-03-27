import tensorflow as tf


# fetch是指在会话中同时运行多个op的操作
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2, input3)
mul = tf.multiply(input1, add)

with tf.Session() as sess:
    result = sess.run([mul, add])
    print(result)

# feed的概念非常重要，指的是在代码初期，先用占位符确定tensor的类型，再在run的时候以字典的形式给占位符赋值
# 定义两个占位符，相当于只是声明变量名和类型，不赋初值
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    # feed的数据以字典的形式传入
    print(sess.run(output, feed_dict={input1: [7], input2: [2]}))
