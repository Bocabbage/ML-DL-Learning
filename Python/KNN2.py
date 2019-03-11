# K-Nearest Neighbors(Classification)
# 更新时间：2019/3/6
#          2019/3/7(修改小错误：为测试集features归一化)
#          2019/3/10(增加PCA[未确定PCA与归一化关系故暂时忽略归一化])
#          2019/3/11(PCA+归一化测试、超参数实验)
# 描述：照搬了腿爷的写法，向腿爷同志学习.jpg         
# 数据形式：np.array
#          train_data(60000*784)
#          train_labels(60000,)
#          test_data(10000*784)
#          test_labels(10000,)
# 结果：
#       归一化未PCA：96.65%
#       PCA未归一化：88.56%(压缩至150)
#       PCA+归一化：92.03%(压缩至150)
#       PCA+归一化：87.07%(压缩至15*15)
import time
import numpy as np 
from scipy.stats import mode
from mlxtend.data import loadlocal_mnist 
from sklearn.decomposition import PCA

def Decomposition(trainX,testX,n_components=150):
    """use PCA to decompose the data"""
    pca = PCA(n_components = n_components,
              svd_solver = "randomized",
              whiten = True).fit(trainX)
    trainX_pca = pca.transform(trainX)
    testX_pca = pca.transform(testX)
    return trainX_pca, testX_pca

#def Binaryzation(data,threshold):
#    res = data
#    res[res<threshold] = 0
#    res[res>=threshold] = 1
#    return res

def CheckResult(test_labels,predict_labels):
    n = test_labels.shape[0]
    correct_l = 0
    for i in range(0,n):
        if test_labels[i] == predict_labels[i]:
            correct_l += 1
    print("test accuracy: %f" %(correct_l/n))
    return

class KNN:
    def __init__(self,k):
        self.k = k

    def fit(self,features,labels):
        self.features = features
        self.labels = labels

    def predict_one_sample(self,feature):
        diff = (self.features - feature)
        dst = np.einsum('ij,ij->i',diff,diff)
        nearest = self.labels[np.argsort(dst)[:self.k]]
        return mode(nearest)[0][0]

    def predict(self,features):
        return np.apply_along_axis(self.predict_one_sample, 1, features)

tr_data,tr_labels = loadlocal_mnist(images_path='E:/MNIST/train-images.idx3-ubyte',
                                    labels_path='E:/MNIST/train-labels.idx1-ubyte') 
ts_data,ts_labels = loadlocal_mnist(images_path='E:/MNIST/t10k-images.idx3-ubyte',
                                    labels_path='E:/MNIST/t10k-labels.idx1-ubyte')


#mtr_data = Binaryzation(tr_data[0:10000,:],127)
#mtr_labels = tr_labels[0:10000]
#mts_data = Binaryzation(ts_data[0:100,:],127)
#mts_labels = ts_labels[0:100]

# 使用PCA进行数据压缩/主成分提取
tr_data_pca,ts_data_pca = Decomposition(tr_data, ts_data,15*15)
tr_data_pca = (tr_data_pca/tr_data_pca.max()).reshape([-1,15*15])
ts_data_pca = (ts_data_pca/ts_data_pca.max()).reshape([-1,15*15])

start_time = time.time()
knn = KNN(k=10)
# 归一化
#knn.fit((mtr_data/mtr_data.max()).reshape([-1,28*28]), mtr_labels)
#predict = knn.predict((mts_data/mts_data.max()).reshape([-1,28*28]))

knn.fit(tr_data_pca, tr_labels)
predict = knn.predict(ts_data_pca)
end_time = time.time()
#CheckResult(mts_labels, predict)
CheckResult(ts_labels, predict)
print("Time:%0.3f s\n"%(end_time-start_time))

