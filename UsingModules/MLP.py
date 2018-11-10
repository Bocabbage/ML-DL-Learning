#项目：使用多层感知机(Multilayer Perceptron,MLP)对MNIST进行分类
#python3.6.6 tensorflow 1.8.0
#Date:2018\10\20
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
tf.enable_eager_execution()

print("Tensorflow version:  {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#用于读取MNIST数据的类#
class DataLoader():
    def __init__(self):
       mnist = input_data.read_data_sets('./data/MNIST_data'\
       ,one_hot=False,source_url='http://yann.lecun.com/exdb/mnist/')
       self.train_data=mnist.train.images                               #np.array [55000,784]
       self.train_labels=np.asarray(mnist.train.labels,dtype=np.int32)  #np.array[55000](int32)
       self.eval_data = mnist.test.images                               #np.array [10000,784]
       self.eval_labels = np.asarray(mnist.test.labels, dtype=np.int32) #np.array[10000](int32)

    def get_batch(self,batch_size):
        index = np.random.randint(0,np.shape(self.train_data)[0],batch_size)
        return self.train_data[index, :], self.train_labels[index]

#多层感知机模型类#
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # 使用Keras的全连接层
        self.dense1 = tf.keras.layers.Dense(units=100,activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self,inputs):
        # 调用次序与返回
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x
    
    def predict(self,inputs):
        # 选择值最大(预测结果)者输出
        logits = self(inputs)
        return tf.argmax(logits,axis=-1)
        # tf.argmax():Returns the index with 
        # the largest value across axes of a tensor. 
        # arg-axis:  Describes which axis of the input Tensor to reduce across.
        # axis 赋值为-1参考 https://tf.wiki/zh/models.html
        
#超参数列表#
num_batches=1000
batch_size=50
learning_rate=0.001

#实例化模型#
model = MLP()
data_loader = DataLoader()
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

#以下迭代完成以下步骤：
#   从DataLoader中随机选取一批训练数据
#   将这批数据送入模型，计算出模型的预测值
#   将模型预测值与labels比较，计算损失函数(loss)
#   计算损失函数关于模型变量的导数
#   使用优化器更新模型参数以最小化损失函数

for batch_index in range(num_batches):
    X,y = data_loader.get_batch(batch_size)
    with tf.GradientTape() as tape:
        y_logit_pred = model(tf.convert_to_tensor(X))
        loss = tf.losses.sparse_softmax_cross_entropy(labels=y,logits=y_logit_pred)
        print("batch %d: loss %f" %(batch_index,loss.numpy())) 
    grads = tape.gradient(loss,model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads,model.variables))

num_eval_samples = np.shape(data_loader.eval_labels)[0]
y_pred = model.predict(data_loader.eval_data).numpy()
print("test accuracy: %f" % \
(sum(y_pred == data_loader.eval_labels) / num_eval_samples))