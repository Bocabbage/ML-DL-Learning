# K-Nearest Neighbors(Classification)
# 更新时间：2019/3/5
# 数据形式：np.array
#          train_data(60000*784)
#          train_labels(60000,)
#          test_data(10000*784)
#          test_labels(10000,)
import time
#import operator
import numpy as np 
from collections import Counter
from mlxtend.data import loadlocal_mnist 

def Binaryzation(data,threshold):
    row = data.shape[0]
    col = data.shape[1]
    res = np.zeros([row,col])
    for i in range(0,row):
        for j in range(0,col):
            if(data[i][j]<threshold):
                res[i][j]=0.0
            else:
                res[i][j]=1
    return res

def SearchNeighbors(k,train_data,train_labels,test_data):
    def Distance(mat,vect):
        sample_nums = mat.shape[0]
        result = []
        dismat = np.tile(vect,(sample_nums,1))-mat
        sqdismat = dismat**2
        sqdistance = sqdismat.sum(axis=1)
        result = sqdistance.argsort()
        #for i in range(0,sample_nums):
        #    result.append((sqdistance[i],i))

        return result


    predict_labels = []
    for i in range(0,test_data.shape[0]):
        res = Distance(train_data, test_data[i,:])
        #res = sorted(res,key=operator.itemgetter(0))
        #k_nearest = [train_labels[x[1]] for x in res[0:k]]
        k_nearest = [train_labels[x] for x in res[0:k]]
        k_counts = Counter(k_nearest)
        predict_labels.append(k_counts.most_common(1)[0][0])
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

mtr_data = Binaryzation(tr_data[0:10000,:],127)
mtr_labels = tr_labels[0:10000]
mts_data = Binaryzation(ts_data[0:100,:],127)
mts_labels = ts_labels[0:100]

start_time = time.time()
pre_labels = SearchNeighbors(k=10,train_data=mtr_data,
                             train_labels=mtr_labels, 
                             test_data=mts_data)
end_time = time.time()
CheckResult(test_labels=mts_labels, 
            predict_labels=pre_labels)

print("Time:%f s\n"%(end_time-start_time))



