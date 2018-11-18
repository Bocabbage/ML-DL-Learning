import time
import tflearn
import tensorflow as tf
import numpy as np
from tflearn import input_data, dropout, fully_connected
from tflearn import conv_2d, max_pool_2d, conv_2d_transpose, upsample_2d
from tflearn import merge
from tflearn import regression
from tflearn import dropout
#from tflearn import ImagePreprocessing
#from tflearn import ImageAugmentation
from tflearn import Momentum
from scipy import misc
from libtiff import TIFF 
#from tflearn.metrics import Accuracy
import matplotlib.pyplot as plt
import matplotlib

import os
import sys
import random
import itertools

IMAGE_H=400
IMAGE_W=400
IMAGE_C=1
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# def tiff_read(tiff_image_name):
# 	tif = TIFF.open(tiff_image_name,mode = 'r')
# 	im_stack = list()
# 	for im in list(tif.iter_images()):
# 		im_stack.append(im)
# 	return im_stack

def Load_BMP_Folder(rootdir):
    List = os.listdir(rootdir)
    path = os.path.join(rootdir,List[0])
    if os.path.isfile(path):
        im_arr = misc.imread(path)
        im_arr = np.reshape(im_arr,(1,IMAGE_H,IMAGE_W))
    for i in range(1,len(List)):
        path = os.path.join(rootdir,List[i])
        if os.path.isfile(path):
            im_temp = misc.imread(path)
            im_temp = np.reshape(im_temp,(1,IMAGE_H,IMAGE_W))
            im_arr = np.concatenate([im_arr,im_temp],0)
    return im_arr

#for i in range(0,30):
#    trainX[i] = sum(trainX[i].tolist(),[])
#    trainY[i] = sum(trainY[i].tolist(),[])
X_rootdir = 'F:\\Seu\\SRTP\\Train\\12.30\\training'
Y_rootdir = 'F:\\Seu\\SRTP\\Train\\12.30\\labels'
trainX = Load_BMP_Folder(X_rootdir)
trainY = Load_BMP_Folder(Y_rootdir)
TrainingsetSize=len(trainX)

# trainX = np.array(tiff_read("train-volume.tiff"))
trainX=np.reshape(trainX,[-1,400,400,1])
# trainY = np.array(tiff_read("train-labels.tiff"))
trainY=np.reshape(trainY,[-1,400,400,1])
trainY = trainY/255.

# 前需读取数据：训练数据trainX,trainY
############ 建立模型 ############
#tf.reset_default_graph()
print(len(trainX))
# 此处需要设定参数：BATCH(?,似乎默认为None),IMAGE_H,IMAGE_W,IMAGE_C(图片高/宽/通道数)
d0 = input_data(shape=[None, IMAGE_H,IMAGE_W, IMAGE_C], name="input")
#data_preprocessing=img_prep,\
#data_augmentation=img_aug)


c1 = conv_2d(d0,  16, 3, weights_init='variance_scaling', activation='prelu', name="conv1_1")
c1 = conv_2d(c1,  16, 3, weights_init='variance_scaling', activation='prelu', name="conv1_2")
p1 = max_pool_2d(c1, 2)

c2 = conv_2d(p1, 32, 3, weights_init='variance_scaling', activation='prelu', name="conv2_1")
c2 = conv_2d(c2, 32, 3, weights_init='variance_scaling', activation='prelu', name="conv2_2")
p2 = max_pool_2d(c2, 2)

c3 = conv_2d(p2, 64, 3, weights_init='variance_scaling', activation='prelu', name="conv3_1")
c3 = conv_2d(c3, 64, 3, weights_init='variance_scaling', activation='prelu', name="conv3_2")
p3 = max_pool_2d(c3, 2)

c4 = conv_2d(p3, 128, 3, weights_init='variance_scaling', activation='prelu', name="conv4_1")
c4 = conv_2d(c4, 128, 3, weights_init='variance_scaling', activation='prelu', name="conv4_2")
c4 = dropout(c4,0.5)
p4 = max_pool_2d(c4, 2)

c5 = conv_2d(p4, 256, 3, weights_init='variance_scaling', activation='prelu', name="conv5_1")
c5 = conv_2d(c5, 256, 3, weights_init='variance_scaling', activation='prelu', name="conv5_2")
c5 = dropout(c5,0.5)



u6 = conv_2d_transpose(c5, 64, 2, [50,50], strides=2)
u6 = merge([u6, c4], mode='concat', axis=3, name='upsamle-5-merge-4')
c6 = conv_2d(u6, 64, 3, weights_init='variance_scaling', activation='prelu', name="conv6_1")
c6 = conv_2d(c6, 64, 3, weights_init='variance_scaling', activation='prelu', name="conv6_2")

u7 = conv_2d_transpose(c6, 32, 2, [ 100, 100], strides=2)
u7 = merge([u7, c3], mode='concat', axis=3, name='upsamle-6-merge-3')
c7 = conv_2d(u7, 32, 3, weights_init='variance_scaling', activation='prelu', name="conv7_1")
c7 = conv_2d(c7, 32, 3, weights_init='variance_scaling', activation='prelu', name="conv7_2")

u8 = conv_2d_transpose(c7, 16, 2, [200, 200], strides=2)
u8 = merge([u8, c2], mode='concat', axis=3, name='upsamle-7-merge-2')        
c8 = conv_2d(u8, 16, 3, weights_init='variance_scaling', activation='prelu', name="conv8_1")
c8 = conv_2d(c8, 16, 3, weights_init='variance_scaling', activation='prelu', name="conv8_2")

u9 = conv_2d_transpose(c8,  8, 2, [400,400], strides=2)
u9 = merge([u9, c1], mode='concat', axis=3, name='upsamle-8-merge-1')
c9 = conv_2d(u9,  16, 3, weights_init='variance_scaling', activation='prelu', name="conv9_1")
c9 = conv_2d(c9,  16, 3, weights_init='variance_scaling', activation='prelu', name="conv9_2")

fc = conv_2d(c9,  1, 1, weights_init='variance_scaling', activation='sigmoid', name="target")

# Define IoU metrics (不是很能看得明白，IOU似乎是Intersection over Union：交并集比)
# 推测是用来评价分割准确率的函数
# def mean_iou_accuracy_op(y_pred, y_true, x):
#     with tf.name_scope('Accuracy'):
#         prec = []
#         for t in np.arange(0.5, 1.0, 0.05):
#             y_pred_tmp = tf.to_int32(y_pred > 0.5)
#             score, update_op = tf.metrics.mean_iou(y_true, y_pred_tmp, 2)
#             with tf.Session() as sess:
#                 sess.run(tf.local_variables_initializer())
#             with tf.control_dependencies([update_op]):
#                 score = tf.identity(score)
#             prec.append(score)
#         acc = tf.reduce_mean(tf.stack(prec), axis=0, name='mean_iou')
#     return acc

#acc=tflearn.metrics.Accuracy()

net = regression(fc, 
                 optimizer='Adam',            # 训练用的Adam算法，但前面import包里出现了Momentum
                 loss='binary_crossentropy',  # 损失函数用的是交叉熵
                 #metric= mean_iou_accuracy_op, # 这里与前文IOU_metrics部分有关 
                 learning_rate=0.001)          # 学习率


model = tflearn.DNN(net,tensorboard_verbose=3) # tensorboard参数设定log内记录值

start_time = time.time()
model.fit(trainX, 
          trainY, 
          validation_set=0.1,   # 表示从training_set中取10%作为交叉验证集
          n_epoch=1,           # 训练轮数
          batch_size=9,        # 每个batch大小
          shuffle=True,         # 是否随机打乱数据
          show_metric=True)     # 是否输出每轮的准确率          
       

duration = time.time() - start_time
print('Training Duration %.3f sec' % (duration))

