import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['PYTHONHASHSEED'] = str(42)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,Concatenate, Conv2DTranspose,GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Add, Activation, Lambda 
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.optimizers import adam_v2
# from keras import backend as keras
from data import *
import tensorflow as tf
import copy
import pdb
import surface_distance as surfdist
# gpus= tf.config.list_physical_devices('GPU') 
# tf.config.experimental.set_memory_growth(gpus[0], True)
np.random.seed(42)
tf.random.set_seed(42)
# tf.config.experimental_run_functions_eagerly(True)
from keras import backend as K
import pdb
import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from keras.activations import sigmoid
from alc import * 
# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N
#-----------------------------------------------------------------------------------------------------------------------------------------------------
# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class labels
    TP = K.sum(y_pred * y_true)
# AUC for a binary classifier
def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)
#-----------------------------------------------------------------------------------------------------------------------------------------------------
def cosine_decay_with_warmup(global_step,
                             learning_rate_base,
                             total_steps,
                             warmup_learning_rate=0.0,
                             warmup_steps=0,
                             hold_base_rate_steps=0):
	"""
	参数：
	global_step: 上面定义的Tcur，记录当前执行的步数。
	learning_rate_base：预先设置的学习率，当warm_up阶段学习率增加到learning_rate_base，就开始学习率下降。
	total_steps: 是总的训练的步数，等于epoch*sample_count/batch_size,(sample_count是样本总数，epoch是总的循环次数)
	warmup_learning_rate: 这是warm up阶段线性增长的初始值
	warmup_steps: warm_up总的需要持续的步数
	hold_base_rate_steps: 这是可选的参数，即当warm up阶段结束后保持学习率不变，知道hold_base_rate_steps结束后才开始学习率下降
	"""
	if total_steps < warmup_steps:
		raise ValueError('total_steps must be larger or equal to '
                         'warmup_steps.')
    #这里实现了余弦退火的原理，设置学习率的最小值为0，所以简化了表达式
	learning_rate = 0.5 * learning_rate_base * (1 + np.cos(np.pi *
        (global_step - warmup_steps - hold_base_rate_steps) / float(total_steps - warmup_steps - hold_base_rate_steps)))
    #如果hold_base_rate_steps大于0，表明在warm up结束后学习率在一定步数内保持不变
	if hold_base_rate_steps > 0:
		learning_rate = np.where(global_step > warmup_steps + hold_base_rate_steps,
                                 learning_rate, learning_rate_base)
	if warmup_steps > 0:
		if learning_rate_base < warmup_learning_rate:
			raise ValueError('learning_rate_base must be larger or equal to '
                             'warmup_learning_rate.')
        #线性增长的实现
		slope = (learning_rate_base - warmup_learning_rate) / warmup_steps
		warmup_rate = slope * global_step + warmup_learning_rate
        #只有当global_step 仍然处于warm up阶段才会使用线性增长的学习率warmup_rate，否则使用余弦退火的学习率learning_rate
		learning_rate = np.where(global_step < warmup_steps, warmup_rate,
                                 learning_rate)
	return np.where(global_step > total_steps, 0.0, learning_rate)

class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    """
    继承Callback，实现对学习率的调度
    """
    def __init__(self,
                 learning_rate_base,
                 total_steps,
                 global_step_init=0,
                 warmup_learning_rate=0.0,
                 warmup_steps=0,
                 hold_base_rate_steps=0,
                 verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.global_step = global_step_init
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.hold_base_rate_steps = hold_base_rate_steps
        self.verbose = verbose
        #learning_rates用于记录每次更新后的学习率，方便图形化观察
        self.learning_rates = []
	#更新global_step，并记录当前学习率
    def on_batch_end(self, batch, logs=None):
        self.global_step = self.global_step + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)
	#更新学习率
    def on_batch_begin(self, batch, logs=None):
        lr = cosine_decay_with_warmup(global_step=self.global_step,
                                      learning_rate_base=self.learning_rate_base,
                                      total_steps=self.total_steps,
                                      warmup_learning_rate=self.warmup_learning_rate,
                                      warmup_steps=self.warmup_steps,
                                      hold_base_rate_steps=self.hold_base_rate_steps)
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nBatch %05d: setting learning '
                  'rate to %s.' % (self.global_step + 1, lr))


class SeBlock(keras.layers.Layer):   
    def __init__(self, reduction=4,**kwargs):
        super(SeBlock,self).__init__(**kwargs)
        self.reduction = reduction
    def build(self,input_shape):#构建layer时需要实现
        #input_shape     
        pass
    def call(self, inputs):
        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(int(x.shape[-1]) // self.reduction, use_bias=False,activation=keras.activations.relu)(x)
        x = keras.layers.Dense(int(inputs.shape[-1]), use_bias=False,activation=keras.activations.hard_sigmoid)(x)
        return keras.layers.Multiply()([inputs,x])    #给通道加权重
        #return inputs*x   


def cbam_block(cbam_feature,ratio=8):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in CBAM: Convolutional Block Attention Module.
	"""
	
	cbam_feature = channel_attention(cbam_feature, ratio)
	cbam_feature = spatial_attention(cbam_feature, )
	return cbam_feature

def spatial_attention(input_feature):
	kernel_size = 7
	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature
	
	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					activation = 'hard_sigmoid',
					strides=1,
					padding='same',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature._keras_shape[-1] == 1
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
		
	return multiply([input_feature, cbam_feature])

def channel_attention(input_feature, ratio=8):
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]
	
	shared_layer_one = Dense(channel//ratio,
							 kernel_initializer='he_normal',
							 activation = 'relu',
							 use_bias=True,
							 bias_initializer='zeros')

	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 bias_initializer='zeros')
	
	avg_pool = GlobalAveragePooling2D()(input_feature)    
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	
	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	
	cbam_feature = Add()([avg_pool,max_pool])
	cbam_feature = Activation('hard_sigmoid')(cbam_feature)
	
	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)
	
	return multiply([input_feature, cbam_feature])

def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0+1e-5))
        # return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def dice_coef_loss(y_true, y_pred):
	return 1 - dice_coef(y_true, y_pred, smooth=1)


class myUnet(object):
	def __init__(self, img_rows = 64, img_cols = 64):
		self.img_rows = img_rows
		self.img_cols = img_cols
# 参数初始化定义
	def load_data(self):
		mydata = dataProcess(self.img_rows, self.img_cols)
		imgs_train, imgs_mask_train = mydata.load_train_data()
		imgs_test = mydata.load_test_data()
		return imgs_train, imgs_mask_train, imgs_test, mydata.test_path
	
	def save_img(self, model, test_path):
		print("array to image")
		volume_dices = []
		hd_dist_95s = []
		avg_surf_dists = []
		surface_overlaps = []
		surface_dices = []
		for a,b,c in os.walk(test_path):
			pass
		for i in c:
			path = os.path.join(test_path,i)
			img = cv2.imread(path,0)
			imgs_test = np.ndarray((1,64,64,1), dtype=np.float32)
			im = img.reshape((64,64,1))
			
			lab = cv2.imread('data/256all/testlab/'+i,0)[96:96+64, 96:96+64]
			
			# imgs_test[0] = (im.astype('float32')/255)**1.5
			imgs_test[0] = im.astype('float32')/255
			# imgs_test[0] = (im/255)**2

			imgs_mask_test = model.predict(imgs_test, verbose=1)
			imgs_mask_test1 = copy.deepcopy(imgs_mask_test)
			lab = lab/255
			# lab = (255-lab)/255
			lab[lab>0.5]=True
			lab[lab<=0.5]=False
			imgs_mask_test1[imgs_mask_test1>0.5]=True
			imgs_mask_test1[imgs_mask_test1<=0.5]=False
			lab = lab.astype(np.bool)
			imgs_mask_test1 = imgs_mask_test1.astype(np.bool)
			# pdb.set_trace()
			#3D-dice值
			volume_dice = surfdist.compute_dice_coefficient(lab, imgs_mask_test1)
			volume_dices.append(volume_dice)
			#豪斯多夫距离
			surface_distances = surfdist.compute_surface_distances(lab, imgs_mask_test1[0][:,:,0], spacing_mm=(1.0, 1.0))
			hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95)
			hd_dist_95s.append(hd_dist_95)
			#平均表面距离
			surface_distances = surfdist.compute_surface_distances(lab, imgs_mask_test1[0][:,:,0], spacing_mm=(1.0, 1.0))
			avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances)
			avg_surf_dists.append(avg_surf_dist)	
			#表面重叠度
			surface_distances = surfdist.compute_surface_distances(lab, imgs_mask_test1[0][:,:,0], spacing_mm=(1.0, 1.0))
			surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1)
			surface_overlaps.append(surface_overlap)
			#表面dice值
			surface_distances = surfdist.compute_surface_distances(lab, imgs_mask_test1[0][:,:,0], spacing_mm=(1.0, 1.0))
			surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1)
			surface_dices.append(surface_dice)
			final_matrix = np.zeros((64, 128,1), np.float32)  
			final_matrix[0:64, 0:64] = imgs_test[0]*255
			final_matrix[0:64, 64:] = imgs_mask_test[0]*255
			cv2.imwrite("result/"+i, imgs_mask_test[0]*255)
			cv2.imwrite("stitch/"+i, final_matrix)
		# pdb.set_trace()
		print("3D-dice值",np.mean(volume_dices))
		print("豪斯多夫距离",np.mean(hd_dist_95s))
		print("平均表面距离",np.mean(avg_surf_dists))
		print("表面重叠度",np.mean(surface_overlaps))
		print("表面dice值",np.mean(surface_dices))


	def get_unet_lca(self):
		inputs = Input((self.img_rows, self.img_cols,1))
		# 网络结构定义
		conv1 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
		print ("conv1 shape:",conv1.shape)
		conv1 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
		print ("conv1 shape:",conv1.shape)
		# conv1 = SeBlock()(conv1)
		mlc1 = Lambda(circ_shift,arguments={'shift':3})(conv1)
		# mlc1 = Lambda(mlc, arguments={'d':[3,5]})(conv1)
		# blam1 = blam_weight()(mlc1)
		x = Conv2D(int(mlc1.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc1)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam1 = Activation('sigmoid')(x)
		# conv1 = cbam_block(conv1)
		
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
		print ("pool1 shape:",pool1.shape)

		conv2 = Conv2D(16*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
		print ("conv2 shape:",conv2.shape)
		conv2 = Conv2D(16*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv2)
		print ("conv2 shape:",conv2.shape)
		drop2 = BatchNormalization()(conv2, training=False)
		conv2 = Activation('relu')(drop2)
		# conv2 = SeBlock()(conv2)
		# conv2 = cbam_block(conv2)
		mlc2 = Lambda(circ_shift,arguments={'shift':3})(conv2)
		# mlc2 = Lambda(mlc, arguments={'d':[3,5]})(conv2)
		# blam2 = blam_weight()(mlc2)
		x = Conv2D(int(mlc2.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc2)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam2 = Activation('sigmoid')(x)

		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
		print ("pool2 shape:",pool2.shape)


		conv3 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
		print ("conv3 shape:",conv3.shape)
		conv3 = Conv2D(32*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv3)
		print ("conv3 shape:",conv3.shape)
		# drop3 = BatchNormalization()(conv3, training=False)
		conv3 = Activation('relu')(conv3)
		conv3 = Dropout(0.5)(conv3)
		# conv3=SeBlock()(conv3)
		# conv3 = cbam_block(drop3)
		mlc3 = Lambda(circ_shift,arguments={'shift':3})(conv3)
		# mlc3 = Lambda(mlc,arguments={'d':[3,5]})(conv3)
		# blam3 = blam_weight()(mlc3)
		x = Conv2D(int(mlc3.shape[-1]) // 4, 1, padding = 'same', kernel_initializer = 'he_normal')(mlc3)
		x = BatchNormalization()(x, training=False)
		x = Activation('relu')(x)
		x = Conv2D(int(x.shape[-1]) * 4, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
		x = BatchNormalization()(x, training=False)
		blam3 = Activation('sigmoid')(x)
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
		print ("pool3 shape:",pool3.shape)
		
		conv4 = Conv2D(64*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
		conv4 = Conv2D(64*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv4)
		# drop4 = Dropout(0.5)(conv4)
		# drop4 = BatchNormalization()(conv4, training=False)
		drop4 = Activation('relu')(conv4)
		drop4 = Dropout(0.5)(drop4)

		up7 = Conv2DTranspose(32*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(drop4)
		# up7 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(drop4))
		up7 = Add()([Multiply()([blam3,up7]), mlc3])
		merge7 = Concatenate(axis=3)([conv3,up7])
		# merge7 = Add()([conv3,up7])
		conv7 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
		conv7 = Conv2D(32*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
		conv7 = Dropout(0.5)(conv7)
		# conv7 = BatchNormalization()(conv7, training=False)
		# conv7 = Activation('relu')(conv7)

		up8 = Conv2DTranspose(16*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(conv7)
		# up8 = Conv2D(16, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(conv7))
		up8 = Add()([Multiply()([blam2,up8]), mlc2])
		merge8 = Concatenate(axis=3)([conv2,up8])
		conv8 = Conv2D(16*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
		conv8 = Conv2D(16*2, 3, padding = 'same', kernel_initializer = 'he_normal')(conv8)
		conv8 = BatchNormalization()(conv8, training=False)
		conv8 = Activation('relu')(conv8)

		up9 = Conv2DTranspose(8*2, 4, activation = 'relu', strides=2, padding='same', kernel_initializer='he_normal')(conv8)
		# up9 = Conv2D(8, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(Conv2DTranspose(size = (2,2))(conv8))
		up9 = Add()([Multiply()([blam1,up9]), mlc1])
		merge9 = Concatenate(axis=3)([conv1,up9])
		conv9 = Conv2D(8*2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
		conv9 = Conv2D(8, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
		conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
		model = Model(inputs = inputs, outputs = conv10)
		# model.compile(optimizer = sgd(lr = 1e-4,momentum=0.9, decay=0.01, nesterov=True), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])
		# model.compile(optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss = [focal_loss(alpha=.25, gamma=5)], metrics = ['accuracy'])
		model.compile(optimizer = adam_v2.Adam(lr=1e-4), loss = [focal_loss(alpha=.25, gamma=2)], metrics = ['accuracy'])#有结果
		# model.run_eagerly = True
		# model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
		return model
# 如果需要修改输入的格式，那么可以从以下开始修改，上面的结构部分不需要修改
	def train(self):
		print("loading data")
		imgs_train, imgs_mask_train, imgs_test, test_path = self.load_data()
		print("loading data done")
		model = self.get_unet_lca()
		print("got unet")
		warmup_batches = 10 * 239 / 4
		total_steps = int(100*239/4)

# Compute the number of warmup batches.
		warmup_steps = int(10 * 239 / 4)

# Create the Learning rate scheduler.
		warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=1e-3,
													total_steps=total_steps,
													warmup_learning_rate=4e-06,
													warmup_steps=warmup_steps,
													hold_base_rate_steps=5,
		)
		model_checkpoint = ModelCheckpoint('lcaunet.h5', monitor='val_loss',verbose=1, save_best_only=True)
		EarlyStop = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=3, verbose=0, mode='min', baseline=None, restore_best_weights=False)
		Reduce = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10,verbose=1,mode='auto',epsilon=0.01,cooldown=10, min_lr=0)
		print('Fitting model...')
		model.summary()
		try:
			model.fit(imgs_train, imgs_mask_train, batch_size=4, epochs=50, verbose=1,validation_split=0.2, shuffle=True, callbacks=[model_checkpoint,warm_up_lr])
		except KeyboardInterrupt:
			print('predict test data')
			# pdb.set_trace()
			val_loss = model.history.history['val_loss']
			loss = model.history.history['loss']
			# accuracy = model.history.history['accuracy']
			# val_accuracy = model.history.history['val_accuracy']
			
			epochs = range(1, len(loss) + 1)
			
			plt.title('val_loss and Loss')
			plt.plot(epochs, val_loss, 'red', label='Validation loss')
			plt.plot(epochs, loss, 'blue', label='Training loss')
			# plt.plot(epochs, accuracy, 'g', label='Validation loss')
			# plt.plot(epochs, val_accuracy, 'y', label='Training loss')

			plt.legend()
			plt.show()
			self.save_img(model, test_path)
			return model, test_path
		# pdb.set_trace()
		print('predict test data')
		val_loss = model.history.history['val_loss']
		loss = model.history.history['loss']
		# accuracy = model.history.history['accuracy']
		# val_accuracy = model.history.history['val_accuracy']
		
		epochs = range(1, len(loss) + 1)
		
		plt.title('val_loss and Loss')
		plt.plot(epochs, val_loss, 'red', label='Validation loss')
		plt.plot(epochs, loss, 'blue', label='Training loss')
		# plt.plot(epochs, accuracy, 'g', label='Validation loss')
		# plt.plot(epochs, val_accuracy, 'y', label='Training loss')
		plt.legend()
		plt.show()
		return model, test_path


	# def save_img(self, model, test_path):
	# 	print("array to image")
			
	# 	for a,b,c in os.walk(test_path):
	# 		pass
	# 	for i in c:
	# 		path = os.path.join(test_path,i)
	# 		img = cv2.imread(path,0)
	# 		imgs_test = np.ndarray((1,64,64,1), dtype=np.float32)
	# 		im = img.reshape((64,64,1))
	# 		# pdb.set_trace()
	# 		imgs_test[0] = (im.astype('float32')/255)**1.5
	# 		# imgs_test[0] = (im/255)**2

	# 		imgs_mask_test = model.predict(imgs_test, verbose=1)
	# 		final_matrix = np.zeros((64, 128,1), np.float32)  
	# 		final_matrix[0:64, 0:64] = imgs_test[0]*255
	# 		final_matrix[0:64, 64:] = imgs_mask_test[0]*255
	# 		cv2.imwrite("result/"+i, imgs_mask_test[0]*255)
	# 		cv2.imwrite("stitch/"+i, final_matrix)

		
if __name__ == '__main__':
	myunet = myUnet()
	model, test_path = myunet.train()
	myunet.save_img(model, test_path)
