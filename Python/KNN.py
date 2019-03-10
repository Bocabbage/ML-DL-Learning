# K-Nearest Neighbors(Classification)
# 更新时间：2019/3/5
#          2019/3/6(修改二值化写法，整体重写见KNN2.py)
#          2019/3/7(更改错误，增加归一化，使test accuracy达到KNN2.py的水平
#                   没有充分利用numpy向量化计算特性，速度比KNN2.py慢一倍)
# 数据形式：np.array
#          train_data(60000*784)
#          train_labels(60000,)
#          test_data(10000*784)
#          test_labels(10000,)
import time
import numpy as np 
from scipy.stats import mode
from mlxtend.data import loadlocal_mnist 
import matplotlib.pyplot as plt

def Binaryzation(data,threshold):
    res = data
    res[res<threshold] = 0
    res[res>=threshold] = 1
    return res

def SearchNeighbors(k,train_data,train_labels,test_data):
    def Distance(mat,vect):
        #sample_nums = mat.shape[0]
        result = []
        dismat = mat-vect
        sqdismat = dismat**2
        sqdistance = sqdismat.sum(axis=1)
        result = sqdistance.argsort()
        #for i in range(0,sample_nums):
        #    result.append((sqdistance[i],i))

        return result


    predict_labels = []
    for i in range(0,test_data.shape[0]):
        res = Distance(train_data, test_data[i,:])
        k_nearest = [train_labels[x] for x in res[0:k]]
        predict_labels.append(mode(k_nearest)[0][0])
    return predict_labels

def CheckResult(test_labels,predict_labels):
    n = test_labels.shape[0]
    correct_l = 0
    for i in range(0,n):
        if test_labels[i] == predict_labels[i]:
            correct_l += 1
    print("test accuracy: %f" %(correct_l/n))
    return


tr_data,tr_labels = loadlocal_mnist(images_path='E:/MNIST/train-images.idx3-ubyte',
                                    labels_path='E:/MNIST/train-labels.idx1-ubyte') 
ts_data,ts_labels = loadlocal_mnist(images_path='E:/MNIST/t10k-images.idx3-ubyte',
                                    labels_path='E:/MNIST/t10k-labels.idx1-ubyte')

#mtr_data = Binaryzation(tr_data[:,:],127)
#mtr_data = tr_data[0:10000,:]
#mtr_labels = tr_labels[0:10000]
#mts_data = Binaryzation(ts_data[:,:],127)
#mts_data = ts_data[0:100,:]
#mts_labels = ts_labels[0:100]

start_time = time.time()
pre_labels = SearchNeighbors(k=10,train_data=(tr_data/(tr_data.max())).reshape([-1,28*28]),
                             train_labels=tr_labels, 
                             test_data=(ts_data/(ts_data.max())).reshape([-1,28*28]))
end_time = time.time()
CheckResult(test_labels=ts_labels, 
            predict_labels=pre_labels)

print("Time:%f s\n"%(end_time-start_time))



