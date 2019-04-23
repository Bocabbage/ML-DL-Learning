

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image
# from sklearn.cluster import SpectralClustering

# Load the picture and visulize it
test_pic = mpimg.imread('E:\\Programming\\Dataset\\Echocardiography\\inp_7.png')
print(test_pic.shape)
#plt.imshow(test_pic)
#plt.show()

# Gray
test_pic_gray = np.dot(test_pic[...,:3],[0.299,0.587,0.114])
print(test_pic_gray.shape)
plt.imshow(test_pic_gray)
plt.show()

# Binarization
# mean = np.mean(test_pic_gray)
# test_pic_bi = test_pic_gray
# test_pic_bi[test_pic_bi<mean] = 0
# test_pic_bi[test_pic_bi>=mean] = 255
# plt.imshow(test_pic_bi)
# plt.show()

# Clustering
test_pic_gray = test_pic_gray.astype(float)

graph = image.img_to_graph(test_pic_gray)
# Note that if the values of your similarity matrix are not well distributed, 
# e.g. with negative values or with a distance matrix rather than a similarity, 
# the spectral problem will be singular and the problem not solvable. 
# In which case it is advised to apply a transformation to the entries of the matrix. 
# For instance, in the case of a signed distance matrix, 
# is common to apply a heat kernel:
# similarity = np.exp(-beta * distance / distance.std())
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph,n_clusters=4).reshape(test_pic_gray.shape)
plt.imshow(labels)
plt.show()