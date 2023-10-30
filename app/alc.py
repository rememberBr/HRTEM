import numpy as np
import keras
from keras.layers import Conv2D, Multiply, Concatenate, minimum, maximum, Lambda, Subtract, Maximum, Minimum
from keras.layers.core import Activation
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
# from keras.backend import minimum, maximum
import pdb

def slice(x,a,b,c,d):
    return x[:,a:b,c:d,:]

def circ_shift(cen, shift):
    # pdb.set_trace()
    _, hei, wid,_ = cen.shape
    
    ######## B1 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B1_NW = Lambda(slice,arguments={'a':shift,'b':None,'c':shift,'d':None})(cen)
    B1_NE = Lambda(slice,arguments={'a':shift,'b':None,'c':None,'d':shift})(cen)
    B1_SW = Lambda(slice,arguments={'a':None,'b':shift,'c':shift,'d':None})(cen)
    B1_SE = Lambda(slice,arguments={'a':None,'b':shift,'c':None,'d':shift})(cen)
    B1_N = Concatenate(axis=2)([B1_NW, B1_NE])
    B1_S = Concatenate(axis=2)([B1_SW, B1_SE])
    B1 = Concatenate(axis=1)([B1_N, B1_S])

    ######## B2 #########
    # old: A  =>  new: B
    #      B  =>       A
    B2_N = Lambda(slice,arguments={'a':shift,'b':None,'c':None,'d':None})(cen)
    B2_S = Lambda(slice,arguments={'a':None,'b':shift,'c':None,'d':None})(cen)
    B2 = Concatenate(axis=1)([B2_N, B2_S])

    ######## B3 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B3_NW = Lambda(slice,arguments={'a':shift,'b':None,'c':wid-shift,'d':None})(cen)
    B3_NE = Lambda(slice,arguments={'a':shift,'b':None,'c':None,'d':wid-shift})(cen)
    B3_SW = Lambda(slice,arguments={'a':None,'b':shift,'c':wid-shift,'d':None})(cen)
    B3_SE = Lambda(slice,arguments={'a':None,'b':shift,'c':None,'d':wid-shift})(cen)
    B3_N = Concatenate(axis=2)([B3_NW, B3_NE])
    B3_S = Concatenate(axis=2)([B3_SW, B3_SE])
    B3 = Concatenate(axis=1)([B3_N, B3_S])

    ######## B4 #########
    # old: AB  =>  new: BA
    B4_W = Lambda(slice,arguments={'a':None,'b':None,'c':wid-shift,'d':None})(cen)
    B4_E = Lambda(slice,arguments={'a':None,'b':None,'c':None,'d':wid-shift})(cen)
    B4 = Concatenate(axis=2)([B4_W, B4_E])

    ######## B5 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B5_NW = Lambda(slice,arguments={'a':hei-shift,'b':None,'c':wid-shift,'d':None})(cen)
    B5_NE = Lambda(slice,arguments={'a':hei-shift,'b':None,'c':None,'d':wid-shift})(cen)
    B5_SW = Lambda(slice,arguments={'a':None,'b':hei-shift,'c':wid-shift,'d':None})(cen)
    B5_SE = Lambda(slice,arguments={'a':None,'b':hei-shift,'c':None,'d':wid-shift})(cen)
    B5_N = Concatenate(axis=2)([B5_NW, B5_NE])
    B5_S = Concatenate(axis=2)([B5_SW, B5_SE])
    B5 = Concatenate(axis=1)([B5_N, B5_S])

    ######## B6 #########
    # old: A  =>  new: B
    #      B  =>       A
    B6_N = Lambda(slice,arguments={'a':hei-shift,'b':None,'c':None,'d':None})(cen)
    B6_S = Lambda(slice,arguments={'a':None,'b':hei-shift,'c':None,'d':None})(cen)
    B6 = Concatenate(axis=1)([B6_N, B6_S])

    ######## B7 #########
    # old: AD  =>  new: CB
    #      BC  =>       DA
    B7_NW = Lambda(slice,arguments={'a':hei-shift,'b':None,'c':shift,'d':None})(cen)
    B7_NE = Lambda(slice,arguments={'a':hei-shift,'b':None,'c':None,'d':shift})(cen)
    B7_SW = Lambda(slice,arguments={'a':None,'b':hei-shift,'c':shift,'d':None})(cen)
    B7_SE = Lambda(slice,arguments={'a':None,'b':hei-shift,'c':None,'d':shift})(cen)
    B7_N = Concatenate(axis=2)([B7_NW, B7_NE])
    B7_S = Concatenate(axis=2)([B7_SW, B7_SE])
    B7 = Concatenate(axis=1)([B7_N, B7_S])

    ######## B8 #########
    # old: AB  =>  new: BA
    B8_W = Lambda(slice,arguments={'a':None,'b':None,'c':shift,'d':None})(cen)
    B8_E = Lambda(slice,arguments={'a':None,'b':None,'c':None,'d':shift})(cen)
    B8 = Concatenate(axis=2)([B8_W, B8_E])
    
    s1 = Multiply()([Subtract()([B1, cen]), Subtract()([B5, cen])])
    s2 = Multiply()([Subtract()([B2, cen]), Subtract()([B6, cen])])
    s3 = Multiply()([Subtract()([B3, cen]), Subtract()([B7, cen])])
    s4 = Multiply()([Subtract()([B4, cen]), Subtract()([B8, cen])])

    # c12 = Minimum()([s1, s2])
    # c123 = Minimum()([c12, s3])
    # c1234 = Minimum()([c123, s4])
    c1234 = Minimum()([s1, s2, s3, s4])
    return c1234

def mlc(cen,d):
    res = []
    for i in d:
        res.append(Lambda(circ_shift,arguments={'shift':i})(cen))
    return Maximum()(res)

class blam_weight(keras.layers.Layer):   
    def __init__(self, reduction=4,**kwargs):
        super(blam_weight,self).__init__(**kwargs)
        self.reduction = reduction
    # def build(self,input_shape):#构建layer时需要实现
    # 	#input_shape     
    # 	pass

    def call(self, inputs):
        # pdb.set_trace()
        x = Conv2D(int(inputs.shape[-1]) // self.reduction, 1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
        x = BatchNormalization()(x, training=False)
        x = Activation('relu')(x)
        x = Conv2D(int(x.shape[-1]) * self.reduction, 1, padding = 'same', kernel_initializer = 'he_normal')(x)
        x = BatchNormalization()(x, training=False)
        x = Activation('sigmoid')(x)
        return x    