from keras.callbacks import ModelCheckpoint
from keras.models import load_model

from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Lambda
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
import tensorflow as tf


import pandas as pd
import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np

from keras.utils import to_categorical
from keras.callbacks import EarlyStopping


import pandas as pd
import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np

import sys
sys.path.insert(0, '../')

from train_gender_classifier_branched import FFFaceDataGenerator

TRAIN_TEST_SPLIT = 0.8
IM_WIDTH = IM_HEIGHT = 198

data_path = '../data/raw/'
image_dir = os.path.join(data_path, 'fairface_pad025')

train_csv = pd.read_csv(os.path.join(data_path, 'fairface_label_train.csv'))
val_csv = pd.read_csv(os.path.join(data_path, 'fairface_label_val.csv'))
df = pd.concat([train_csv, val_csv])


dataset_dict = {
    'gender_id': {
        0: 'Male',
        1: 'Female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())

                
data_generator = FFFaceDataGenerator(df)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("../model_checkpoint/gender-cp-cont-{epoch:04d}.hdf5", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=False, 
    save_weights_only=False, 
    mode='auto', 
    period=1),
    EarlyStopping(monitor='loss', patience=20)
]

checkpoint_path = "../model_checkpoint/gender-cp-0045.hdf5"
initial_epoch = 55

model = load_model(checkpoint_path) 

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=100,
                    callbacks=callbacks,
                    initial_epoch=initial_epoch,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


