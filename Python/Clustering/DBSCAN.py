# DBSCAN Algorithm
# 更新时间：2019/4/3

import numpy as np
import random

class DBSCAN:
    """
        DBSCAN(Density-Based Spatial Clustering of 
               Applications with Noise)

        Note: Defaultly use the Eular-Distance
    """
    def __init__(self,eps,min_samples,random_state=0):
    """
        eps:float,optional
            The radius of eps-neighborhood
        min_samples:int,optional
            the MinPts which is the threshold of eps-neighborhood
    """
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state

    def fit(self,X):
    """
        X:nparray
          dimension is [num_samples,features]
    """
        random.seed(self.random_state)
        n_samples = X.shape[0]
        cluster_counts = 0

        # Calculate the distances from points to points
        # (space use can be optimized)
        dsts = np.zeros([n_samples,n_samples],dtype=float)
        for i in range(0,n_samples):
            diff = X - X[i,:]
            dst = np.einsum('ij,ij->i',diff,diff)
            dsts[i,:] = dst

        # record whether the point is visited
        # 0:unvisited <-> 1:visited
        is_visit = np.zeros([n_samples,],dtype=int)
        # Clustering result while >0 means one of the cluster
        # and -1 means 'noise'
        y_predict = np.zeros([n_samples,], dtype=int)

        while 0 in is_visit:
            p = random.randint(0,n_samples)
            while is_visit[p]==1:
                p = random.randint(0, n_samples)
            
            is_visit[p] = 1
            # p's eps neighborhood
            N = []
            for i in range(0,n_samples):
                if dsts[i,p] <= self.eps:
                    N.append(i)
            N.remove(p)
            if((len(N)+1)>=self.min_samples):
                cluster_counts += 1
                # p(regarded as a core object) generates a new cluster
                y_predict[p] = cluster_counts
                for pp in N:
                    if is_visit[pp] == 0:
                        is_visit[pp] == 1
                        NN = []
                        for i in range(0,n_samples):
                            if dst[i,pp] <= self.eps
                            NN.append(i)
                        NN.remove(p)
                        if((len(NN)+1)>=self.min_samples):
                            N = N + NN
                    if y_predict[pp] == 0:
                        y_predict[pp] = cluster_counts
            else:
                y_predict[p] = -1
        self.labels = y_predict

    def fit_predict(self,X):
        self.fit(X)
        return self.labels

    
