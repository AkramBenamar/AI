# This Python file uses the following encoding: utf-8
import tensorflow as tf
import numpy as np
import tarfile
import nibabel as nib
import glob
import time
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sys import stdout
# import matplotlib.pyplot as plt
# import matplotlib.image as mpim
# from scipy.ndimage.interpolation import affine_transform
# from sklearn.model_selection import train_test_split
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D,Conv2D, Conv3DTranspose, Dropout, ReLU, LeakyReLU, Concatenate, ZeroPadding3D, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers import concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
#MODELS Unet3DGan
EXEC = 1
kze=4
VERSION = 'eval_brats21'
INPUT_PATH=""
#MODELS Unet3DGan

#MODELS Unet3DGan
def double_conv(layer, Nf, ks, norm=True):
      x=layer
      del layer
      for ss in range(2):
          x = Conv2D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
          
          #a chaque etape effectuer la convolution afin d obtenir shape/2 avec nombre de filtre comme 4 eme D
          if (norm):
              x = BatchNormalization()(x) #Normaliser l image seulement 
          x = ReLU()(x) #Activer la sortie
      return x   

def Generator():
    '''
    Generator model
    '''
    
    def encoder_step(layer, Nf, ks, norm=True):
        x = double_conv(layer, Nf/2, ks, norm)
        del layer
        x = Conv2D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(x)  
        x = Dropout(0.1)(x) #met a zero 0.2 des donnees afin d eviter l overfiting

        return x

    def bottlenek(layer, Nf, ks):
        x = Conv2D(Nf, kernel_size=ks, strides=2, kernel_initializer='he_normal', padding='same')(layer)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #chaque etape de convolution est concatene avec la precedente (ici prblm de size)
        for i in range(4):
            y = Conv2D(Nf, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(y)
            x = ReLU()(x)
            x = Concatenate()([x, y])
#             print(x.shape)
        return x

    def decoder_step(layer, layer_to_concatenate, Nf, ks):
        x = Dropout(0.1)(layer)
        del layer
        x = Conv2DTranspose(Nf, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal')(x)
        x = Concatenate()([x, layer_to_concatenate])
        x = double_conv(x, Nf, ks)
        return x

    layers_to_concatenate = []
    inputs = Input((512,512,1), name='input_image')
    Nfilter_start = 24
    depth = 5
    ks = 3
    x = inputs

    # encoder
    for d in range(depth):
        if d==0: #les images sont initialement normalisé en prétraitement donc norm=false
            x = Conv2D(Nfilter_start, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
            x = Conv2D(Nfilter_start, kernel_size=ks, strides=1, kernel_initializer='he_normal', padding='same')(x)
        else:
            x = encoder_step(x, Nfilter_start*np.power(2,d), ks)
        
        layers_to_concatenate.append(x)

    # bottlenek
    x = bottlenek(x, Nfilter_start*np.power(2,depth-1), ks)
    
    # decoder
    for d in range(depth-2, -1, -1): 
        x = decoder_step(x, layers_to_concatenate.pop(), Nfilter_start*np.power(2,d), ks)
    
    # classifier
    last = Conv2DTranspose(2, kernel_size=ks, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)
    del layers_to_concatenate,x
    return Model(inputs=inputs, outputs=last, name='Generator')


G = Generator()
# D = Discriminator()

G.load_weights('weights/covidmodel.h5')
print('G Loaded succ')
G.save('covG.h5')
