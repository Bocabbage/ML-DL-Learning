import tensorflow as tf 
import numpy as np
tf.enable_eager_execution()
#tf.enable_eager_execution():
#Eager execution provides an imperative interface to
#TensorFlow. With eager execution enabled, 
#TensorFlow functions execute operations immediately
#(as opposed to adding to a graph to be executed later
#in a tf.Session)
a=tf.constant(1)
b=tf.constant(1)
c=tf.add(a,b)
print(c)

A=tf.constant([[1,2],[3,4]])
B=tf.constant([[5,6],[7,8]])
C=tf.matmul(A,B)
print(C)

x=tf.get_variable('x',shape=[1],initializer=tf.constant_initializer(3.))
#tf.get_variable():
#Gets an existing variable with 
#these parameters or create a new one.
#变量与普通张量的一个重要区别是
#其默认能够被TensorFlow的自动求导机制所求导，
#因此往往被用于定义机器学习模型的参数。
with tf.GradientTape() as tape:
    y1=tf.square(x)
y_grad=tape.gradient(y1,x)
print([y1.numpy(),y_grad.numpy()])


X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.get_variable('w', shape=[2, 1], initializer=tf.constant_initializer([[1.], [2.]]))
b = tf.get_variable('b', shape=[1], initializer=tf.constant_initializer([1.]))
with tf.GradientTape() as tape:
    L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
w_grad, b_grad = tape.gradient(L, [w, b]) # 计算L(w, b)关于w, b的偏导数
print([L.numpy(), w_grad.numpy(), b_grad.numpy()])

#使用常规的科学计算库实现机器学习模型有两个痛点：
#(1)经常需要手工求函数关于参数的偏导数
#(2)经常需要手工根据求导的结果更新参数

#(1)可用tape.gradient()自动计算梯度解决
#(2)可用optimizer.apply_gradients()自动更新参数解决
#TensorFlow的 Eager Execution（动态图）模式
# 与NumPy的运行方式十分类似，
# 然而提供了更快速的运算（GPU支持）、自动求导、优化器等一系列
# 对深度学习非常重要的功能。

#Linear Regression#

#定义数据，进行基本归一化操作
X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)
X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

X=tf.constant(X)
y=tf.constant(y)
a=tf.get_variable('a',dtype=tf.float32,shape=[],\
initializer=tf.zeros_initializer)
b=tf.get_variable('b',dtype=tf.float32,shape=[],\
initializer=tf.zeros_initializer)
variables=[a,b]

num_epoch=10000
optimizer= tf.train.GradientDescentOptimizer(learning_rate=1e-3)
#tf.train.GradientDescentOptimizer():（优化器）
#Optimizer that implements the gradient descent algorithm.
for e in range(num_epoch):
    with tf.GradientTape() as tape:
        y_pred = a*X+b
        loss = 0.5*tf.reduce_sum(tf.square(y_pred-y))
    grads = tape.gradient(loss,variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,\
    variables))
    #print([a.numpy(),b.numpy()])

