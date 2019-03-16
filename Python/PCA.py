# 利用numpy实现PCA
# 更新时间：2019/3/12
#          2019/3/13(Debug,处理中心化)
#          2019/3/14(SVD方案，未处理完全)
#          2019/3/16(SVD方法完成，PCA[用于图像压缩]轮子造完)
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.utils.extmath import svd_flip

class PCA_2d:
    """Take the data forms as X(m*n) to do PCA"""
    def __init__(self,n_components,use_svd='eig'):
        """Let the n-dim data to be n_components-dim"""
        self.n_components = n_components
        self.use_svd = use_svd
    def fit(self,X):
        """X is a 2-d numpy array"""
        # Actually, the 'eig' method and the svd method
        # are the same thing
        mean_X = np.mean(X,axis = 0) 
        X_n = X - mean_X
        if self.use_svd == 'eig':
            sigma = np.dot(X_n.T,X_n)
            #sigma = np.cov(X)
            eigvalue,eigvector = np.linalg.eig(sigma)
            eigvalue = np.real(eigvalue)
            eigvector = np.real(eigvector)
            self.w = eigvector[np.argsort(eigvalue)[self.n_components:],:]
        elif self.use_svd == 'svd':
            # Don't forget the 'full_matrices'=False
            # as it's for reduced SVD
            U,S,V = np.linalg.svd(X_n,full_matrices=False)
            # flip eigenvectors' sign 
            #to enforce deterministic output(from:sklearn/pca.py)
            U,V = svd_flip(U, V)
            self.w = V[:self.n_components]


    def transform(self,Y):
        return np.dot(Y,self.w.T)

# Test
tr_data,tr_labels = loadlocal_mnist(images_path='E:/MNIST/train-images.idx3-ubyte',
                                    labels_path='E:/MNIST/train-labels.idx1-ubyte') 
nc = 15
pca = PCA_2d(nc*nc,"svd")
pca.fit(tr_data[0:1000,:])

pca_sk = PCA(n_components=nc*nc,svd_solver="full")
pca_sk.fit(tr_data[0:1000,:])

tr_data_pca = pca.transform(tr_data[0:1000,:])
#tr_data_pca = pca_sk.transform(tr_data[0:1000,:])
trpic = (tr_data_pca[0,:]).reshape(nc,nc)
plt.matshow(trpic)
plt.show()