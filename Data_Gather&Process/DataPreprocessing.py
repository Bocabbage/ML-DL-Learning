# Coding in utf-8
# For Image in DICOM file reading and cropping
# Processing in batches
# 2018/11/16
import os
import SimpleITK as sitk
import numpy as np
from PIL import Image
# from matplotlib import pyplot as plt
def Load_DCM_File(filename):
	ds = sitk.ReadImage(filename)
	img_array = sitk.GetArrayFromImage(ds)
	img_array = img_array[0]
	return img_array					# 返回读出的DICOM文件图片(ndarray)

rootdir = 'F:\\Seu\\SRTP\\Train\\12.30\\image'
nrootdir = 'F:\\Seu\\SRTP\\Train\\12.30\\training'
List = os.listdir(rootdir)				# 列出文件夹下所有的目录与文件
for i in range(0,len(List)):			# 遍历文件
	path = os.path.join(rootdir,List[i])
	if os.path.isfile(path):
		image = Load_DCM_File(path)		# 读出图像为434*636*3
		image = image[34:434,100:500,:]	# 裁剪为400*400*3
		image = np.array(image,dtype='uint8')
		image = Image.fromarray(image)
		n_filename = List[i][:-3] + 'bmp'
		n_path = os.path.join(nrootdir,n_filename)
		image.save(n_path,'bmp')


