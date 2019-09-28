from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from keras.layers.pooling import AveragePooling3D, MaxPooling2D,MaxPooling3D 
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape,Conv3D 
from keras.layers import LSTM, Dense, TimeDistributed, UpSampling2D, UpSampling3D 
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt
from keras.layers.pooling import AveragePooling3D, MaxPooling3D,MaxPooling2D 
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Reshape 
from keras.layers import LSTM, Dense, TimeDistributed, UpSampling2D, UpSampling3D 
from keras import optimizers 
from keras.models import load_model
from keras.layers import LeakyReLU 
from keras.layers import Dropout
from keras.models import Sequential, Model
from keras.layers.merge import concatenate



def load_fb_trackingmodel(): 
    input_shape=(1, 360, 640, 3)
    input_img=Input(input_shape, name='input')
    
    a0=(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape, activation='relu'))(input_img)   #360
    x=BatchNormalization()(a0) 
    a1=(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape, activation='relu'))(x) 
    x=BatchNormalization()(a1)
    a2=(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape, activation='relu'))(x) 
    x=BatchNormalization()(a2)
    a3=(Conv3D(32, (3, 3, 3), padding='same', input_shape=input_shape, activation='relu'))(x) 
    x=BatchNormalization()(a3)

    d0=(MaxPooling3D(pool_size=(1,2, 2), strides = (2,2,2)))(x)    #180
    d1= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(d0)  
    x=BatchNormalization()(d1) 
    d2= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(d2)
    d3= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(d3)

    x=(MaxPooling3D(pool_size=(1,2, 2), strides = (2,2,2)))(x) #90
    c1= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(c1)  
    c2= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(c2)
    c3= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(c3)

    x=(MaxPooling3D(pool_size=(1,2, 2), strides = (2,2,2)))(x) #45
    x= (Conv3D(256, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)  
    x= (Conv3D(256, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x) 
    x= (Conv3D(256, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)

    x=(UpSampling3D(size=(1, 2,2)))(x) #90
    x=concatenate([c3,x])
    x= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)
    x=concatenate([c2,x])
    x= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)
    x=concatenate([c1,x])
    x= (Conv3D(128, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)


    x=(UpSampling3D(size=(1, 2,2)))(x) #180
    x=concatenate([d3,x])
    x= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x) 
    x=concatenate([d2,x])
    x= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)
    x=concatenate([d1,x])
    x= (Conv3D(64, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)


    x=(UpSampling3D(size=(1, 2,2)))(x) #360
    x=concatenate([a3,x])
    x= (Conv3D(16, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x) 
    x=concatenate([a2,x])
    x= (Conv3D(16, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)
    x=concatenate([a1,x])
    x= (Conv3D(32, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x)
    x= (Conv3D(16, (3, 3, 3), padding='same', activation='relu'))(x)  
    x=BatchNormalization()(x) 
    x= (Conv3D(1, (3, 3, 3), padding='same', activation='sigmoid'))(x)  
    model = Model(input_img,output=[x])
    model.summary()
    return model

model=load_fb_trackingmodel()
model.load_weights('') # Add model weights path. Can be downloaded @ https://storage.cloud.google.com/ball-tracking-model-weights/soccer_ball_tracking_model_weights.h5

