from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np 
import os
import glob
import cv2
# from libtiff import TIFF
import pdb
# import pandas as pd

# def save_csv(name,data,lis):
# 	test=pd.DataFrame(columns=lis,data=data)
# 	test.to_csv(name,encoding='gbk')

class dataProcess(object):
	def __init__(self, out_rows, out_cols, data_path = r"data/256all/train/images", label_path = r"data/256all/train/label", test_path = r"data/256all/test", npy_path = "npydata", img_type = "png"):
		# 数据处理类，初始化
		self.out_rows = out_rows
		self.out_cols = out_cols
		self.data_path = data_path
		self.label_path = label_path
		self.img_type = img_type
		self.test_path = test_path
		self.npy_path = npy_path

# 创建训练数据
	def create_train_data(self):
		i = 0
		print('-'*30)
		print('Creating training images...')
		print('-'*30)
		imgs = glob.glob(self.data_path+"/*."+self.img_type)
		print(len(imgs))

		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		imglabels = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		for imgname in imgs:
			# pdb.set_trace()
			midname = imgname[imgname.rindex("\\")+1:]
			img = load_img(self.data_path + "/" + midname,color_mode = "grayscale")
			label = load_img(self.label_path + "/" + midname,color_mode = "grayscale")
			img = img_to_array(img)
			label = img_to_array(label)
			imgdatas[i] = img
			imglabels[i] = label
			if i % 100 == 0:
				print('Done: {0}/{1} images'.format(i, len(imgs)))
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_train.npy', imgdatas)
		np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
		# pdb.set_trace()
		print('Saving to .npy files done.')

# 创建测试数据
	def create_test_data(self):
		i = 0
		print('-'*30)
		print('Creating test images...')
		print('-'*30)
		imgs = glob.glob(self.test_path+"/*."+self.img_type)
		print(len(imgs))
		imgdatas = np.ndarray((len(imgs),self.out_rows,self.out_cols,1), dtype=np.uint8)
		# pdb.set_trace()
		for imgname in imgs:
			midname = imgname[imgname.rindex("\\")+1:]
			# img = load_img(self.test_path + "/" + midname,grayscale = True)
			img = load_img(self.test_path + "/" + midname,color_mode = "grayscale")
			img = img_to_array(img)
			#img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
			#img = np.array([img])
			imgdatas[i] = img
			i += 1
		print('loading done')
		np.save(self.npy_path + '/imgs_test.npy', imgdatas)
		print('Saving to imgs_test.npy files done.')

# 加载训练图片与mask
	def load_train_data(self):
		print('-'*30)
		print('load train images...')
		print('-'*30)
		imgs_train = np.load(self.npy_path+"/imgs_train.npy")
		imgs_mask_train = np.load(self.npy_path+"/imgs_mask_train.npy")
		imgs_train = imgs_train.astype('float32')
		imgs_mask_train = imgs_mask_train.astype('float32')
		imgs_train /= 255
		# imgs_train = imgs_train**(1/2.5)
		# mean = imgs_train.mean(axis = 0)
		# imgs_train -= mean
		imgs_mask_train /= 255
		# pdb.set_trace()
        # 做一个阈值处理，输出的概率值大于0.5的就认为是对象，否则认为是背景
		imgs_mask_train[imgs_mask_train > 0.5] = 1
		imgs_mask_train[imgs_mask_train <= 0.5] = 0
		# pdb.set_trace()
		return imgs_train,imgs_mask_train

# 加载测试图片
	def load_test_data(self):
		print('-'*30)
		print('load test images...')
		print('-'*30)
		imgs_test = np.load(self.npy_path+"/imgs_test.npy")
		imgs_test = imgs_test.astype('float32')
		imgs_test /= 255
		# imgs_test = imgs_test**2.5
		# mean = imgs_test.mean(axis = 0)
		# imgs_test -= mean
		return imgs_test


if __name__ == "__main__":

	mydata = dataProcess(64,64)
	mydata.create_train_data()
	mydata.create_test_data()

	imgs_train,imgs_mask_train = mydata.load_train_data()
	print (imgs_train.shape,imgs_mask_train.shape)
