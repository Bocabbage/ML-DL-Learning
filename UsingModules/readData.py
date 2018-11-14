from libtiff import TIFF
from PIL import Image

def tiff_read(tiff_image_name):
	tif = TIFF.open(tiff_image_name,mode = 'r')
	im_stack = list()
	for im in list(tif.iter_images()):
		im_stack.append(im)
	return im_stack

def tiff_write(tiff_image_name,im_array,image_num):
	tif = TIFF.open(tiff_image_name,mode = 'w')
	for i in range(0,image_num):
		im = Image.fromarray(im_array[i])
		# 缩放到同一尺寸
		im = im.resize((size1,size2),Image.ANTIALIAS)
		tif.write_image(im,compression = None)
	out_tiff.close()
	return