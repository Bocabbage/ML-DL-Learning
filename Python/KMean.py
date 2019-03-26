# K-Mean Algorithm
# 更新时间:2019/3/26(已完成，未验证)
import numpy as np
import random

class KMean:
    """K-Mean Algorithm for Cluster Analysis"""
    def __init__(self,n_clusters,random_state):
        """n_clusters is the number of clusters"""
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self,X):
        """
        X is the sample matric(i.e. a numpy array)
        whose dim is [n_samples * n_features]

        the fit() method will return a numpy array
        y_predict [n_samples * 1]

        """
        n_samples = X.shape[0]
        random.seed(self.random_state)

        # initial the first n centroids
        centroids = np.zeros([X.shape[1],self.n_clusters])
        index_centroids = []
        for i in range(0,self.n_clusters):
            temp = random.randint(0, n_samples)
            while(temp in index_centroids):
                temp = random.randint(0, n_samples)
            index_centroids.append(temp)
            centroids[:,i]=X[index_centroids,:]


        # Clustering
        y_predict = np.zeros([n_samples,], dtype=int)
        oldy_predict = np.ones([n_samples,],dtype=int)
        Eular_dst = np.zeros([self.n_clusters,n_samples])
        while not (oldy_predict == y_predict).all():
            oldy_predict = y_predict
            for i in range(0,self.n_clusters):
                centroid = centroids[:,i]
                diff = X - centroid
                dst = np.einsum('ij,ij->i',diff,diff)
                Eular_dst[i,:] = dst
            y_predict = np.argmin(Eular_dst,axis=1)
            # Find new centroids
            centroids = centroids - centroids   # re-initial the centroids
            n_cluster_samples = np.zeros([self.n_clusters,], dtype=int)
            for i in range(0,n_samples):
                y_pre = y_predict[i]
                n_cluster_samples[y_pre] += 1
                centroids[:,y_pre] += X[i,:]
            centroids = centroids / n_cluster_samples

        return y_predict
