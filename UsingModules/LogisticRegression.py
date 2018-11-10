#项目：使用LogisticRegression对MNIST进行分类
#python3.6.6
#Date:2018\11\5
import time
import numpy as np
import matplotlib.pyplot as plt 
#from sklearn.datasets import fetch_mldata
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from tensorflow.examples.tutorials.mnist import input_data
#print(__doc__)
t0 = time.time()

#加载数据集：包括训练集与验证集
mnist=input_data.read_data_sets('./data/MNIST_data'\
       ,one_hot=False,source_url='http://yann.lecun.com/exdb/mnist/')

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels,dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
train_samples = train_data.shape[0]


# 训练集的随机打乱
random_state=check_random_state(0)
permutation=random_state.permutation(train_data.shape[0])
train_data=train_data[permutation]
train_labels=train_labels[permutation]

# 数据集的归一化
scaler=StandardScaler()
train_data=scaler.fit_transform(train_data)
eval_data=scaler.transform(eval_data)

# 模型产生与训练
clf = LogisticRegression(C=50./train_samples,\
                         multi_class='multinomial',\
                         penalty='l1',\
                         solver='saga',tol=0.1)
clf.fit(train_data,train_labels)

# 模型验证
# coef_:分类器向量系数
# score:测试集上准确率得分
sparsity=np.mean(clf.coef_==0)*100      
score=clf.score(eval_data,eval_labels)  # 测试集上得分
print("Sparsity with L1 penalty: %.2f%%" % sparsity)
print("Test score with L1 penalty: %.4f" % score)

# 打印提取到的特征图
coef = clf.coef_.copy()
plt.figure(figsize=(10,5))
scale = np.abs(coef).max()
for i in range(10):
    l1_plot = plt.subplot(2,5,i+1)
    l1_plot.imshow(coef[i].reshape(28,28),interpolation='nearest',\
                   vmin=-scale,vmax=scale)
    l1_plot.set_xticks(())
    l1_plot.set_yticks(())
    l1_plot.set_xlabel('Class %i' % i)
plt.suptitle('Classification vector for...')
run_time = time.time() - t0
print('Example run in %.3f s' % run_time)
plt.show()
print('stop!')

