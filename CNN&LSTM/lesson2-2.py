import tensorflow as tf


x = tf.Variable([1, 2])
a = tf.constant([3, 3])

# 定义一个减法op
sub = tf.subtract(x, a)
# 定义一个加法op
add = tf.add(x, sub)

# tf中只要出现变量，就要定义全局变量初始化op
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sub)
    print(add)
    print(sess.run(sub))
    print(sess.run(add))


# 定义一个变量tensor，初始化，并起名字
state = tf.Variable(0, name='counter')

# 创建一个op，使state加一
new_value = tf.add(state, 1)

# 创建一个赋值op，在tf中，tensor赋值不能使用等号，需要调用这个方法
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))

    for _ in range(5):
        # 每次循环执行一次update，update是一个op，会输出一个新的tensor，这个新tensor就赋值给了state
        sess.run(update)
        print(sess.run(state))
        # 注意，这里写成print(sess.run(update))，结果是一样的
