# 利用numpy实现PCA
# 更新时间：2019/3/12
import numpy as np
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt

class PCA_2d:
    """Take the data forms as X(m*n) to do PCA"""
    def __init__(self,n_components):
        """Let the n-dim data to be n_components-dim"""
        self.n_components = n_components
    def fit(self,X):
        """X is a 2-d numpy array"""
        sigma = np.dot(X.T,X)
        eigvalue,eigvector = np.linalg.eig(sigma)
        self.w = eigvector[np.argsort(eigvalue)[-(self.n_components+1):-1],:]
    def transform(self,Y):
        return np.dot(Y,self.w.T)

# Test
tr_data,tr_labels = loadlocal_mnist(images_path='E:/MNIST/train-images.idx3-ubyte',
                                    labels_path='E:/MNIST/train-labels.idx1-ubyte') 
nc = 15
pca = PCA_2d(nc*nc)
pca.fit(tr_data)

tr_data_pca = pca.transform(tr_data)
#plt.matshow((tr_data_pca[0,:]).reshape(nc,nc))
#plt.show()