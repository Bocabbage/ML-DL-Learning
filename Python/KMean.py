# K-Mean Algorithm
# 更新时间:2019/3/26(已完成，未验证)
#         2019/4/1(验证，Unevenly sized blobs存在问题，待解决)
import numpy as np
import random

class KMeans:
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
            centroids[:,i]=X[temp,:]


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
            y_predict = np.argmin(Eular_dst,axis=0)
            # Find new centroids
            centroids = centroids - centroids   # re-initial the centroids
            n_cluster_samples = np.zeros([self.n_clusters,], dtype=int)
            for i in range(0,n_samples):
                y_pre = y_predict[i]
                n_cluster_samples[y_pre] += 1
                centroids[:,y_pre] += X[i,:]
            centroids = centroids / n_cluster_samples

        self.labels_ = y_predict
        self.cluster_centers_ = centroids
        return y_predict

# For test
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs

    plt.figure(figsize=(12,12))

    n_samples = 1500
    random_state = 170
    X,y = make_blobs(n_samples=n_samples,
                     random_state=random_state)
    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit(X)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")

    # Anisotropiclly distributed data
    transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
    X_aniso = np.dot(X, transformation)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit(X_aniso)
    plt.subplot(222)
    plt.scatter(X_aniso[:, 0], X_aniso[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")

    # Different variance
    X_varied, y_varied = make_blobs(n_samples=n_samples,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=random_state)
    y_pred = KMeans(n_clusters=3, random_state=random_state).fit(X_varied)

    plt.subplot(223)
    plt.scatter(X_varied[:, 0], X_varied[:, 1], c=y_pred)
    plt.title("Unequal Variance")

    # Unevenly sized blobs
    X_filtered = np.vstack((X[y == 0][:500], X[y == 1][:100], X[y == 2][:10]))
    y_pred = KMeans(n_clusters=3,
                    random_state=random_state).fit(X_filtered)

    plt.subplot(224)
    plt.scatter(X_filtered[:, 0], X_filtered[:, 1], c=y_pred)
    plt.title("Unevenly Sized Blobs")
    
    plt.show()