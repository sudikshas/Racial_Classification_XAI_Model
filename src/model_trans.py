import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten, ZeroPadding2D
from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense, Convolution2D
from tensorflow.keras import Sequential
#from tensorflow.keras.applications.xception import preprocess_input
#from tensorflow.keras.applications import Xception
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import mobilenet_v2

def build_model(num_classes, size = 224):
    
    model = Sequential()
    
    tr = ResNet50V2(include_top = False,
                       weights = "imagenet",
                       input_shape = (size, size, 3))
        
    for layer in tr.layers[:14]:
        layer.trainable = False
        model.add(layer)
    
    #model.add(keras.Input(shape = (size, size, 3)))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    #model.add(layers.LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    #model.add(Conv2D(64, (3, 3), padding = "same"))
    #model.add(Conv2D(64, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    #model.add(layers.LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    #model.add(layers.LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2)))

    #model.add(Conv2D(128, (3, 3), padding = "same"))
    #model.add(Conv2D(128, (3, 3), padding = "same"))
    model.add(Conv2D(256, (3, 3), padding = "same"))
    model.add(Conv2D(256, (3, 3), padding = "same"))
    model.add(BatchNormalization(axis = -1))
    model.add(layers.Activation(activations.relu))
    #model.add(layers.LeakyReLU())
    model.add(MaxPooling2D(pool_size = (2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation = "relu"))
    #model.add(layers.LeakyReLU())
    model.add(Dropout(0.7))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(256, activation = "relu"))
    #model.add(layers.LeakyReLU())
    #model.add(Dropout(0.3))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(128, activation = "relu"))
    #model.add(layers.LeakyReLU())
    #model.add(Dropout(0.3))
    model.add(BatchNormalization(axis = -1))
    model.add(Dense(num_classes, activation = "softmax"))
    
    return model

"""
class OutputModel():
    '''
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    '''
    def make_default_hidden_layers(self, size, num_ages = 7):
        '''
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> activation -> maxpool
        '''
        
        incept = InceptionV3(include_top = False, 
                          weights = "imagenet",
                          input_shape = (size, size, 3))
        model = Sequential()
        for layer in incept.layers[:18]:
            layer.trainable = False
            model.add(layer)
        

        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        #model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        #model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        #model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(128, (3, 3), padding = "same"))
        model.add(BatchNormalization())
        model.add(layers.Activation(activations.relu))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_ages, activation = "softmax"))
        
        

        return model
"""


class OutputModel():
    '''
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    '''
    def make_default_hidden_layers(self, resize, num_ages = 9):
        '''
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        '''
        
        xcep = Xception(include_top = False,weights = "imagenet",input_shape = (resize, resize, 3))
        model = Sequential()
        for layer in xcep.layers[:12]:
            layer.trainable = False
            model.add(layer)
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        
        model.add(Conv2D(64, (3, 3), padding = "same"))
        model.add(layers.Activation(activations.relu))
        model.add(BatchNormalization(axis = -1))
        model.add(MaxPooling2D(pool_size = (2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(256, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(128, activation = "relu"))
        model.add(Dropout(0.5))
        model.add(BatchNormalization())
        model.add(Dense(num_ages, activation = "softmax"))

        return model

