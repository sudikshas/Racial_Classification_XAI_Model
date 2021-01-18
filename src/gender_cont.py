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


import pandas as pd
import os
from PIL import Image
from tensorflow import keras
import tensorflow as tf
import numpy as np

TRAIN_TEST_SPLIT = 0.7
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

### Preprocess Images

class FFFaceDataGenerator():
    """
    Data generator for the FFFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df):
        self.df = df
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
#         self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

#         self.max_age = self.df['age'].max()
        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((IM_WIDTH, IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, genders = [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                gender = person['gender_id']
                file = os.path.join(image_dir, person['file'])
                
                im = self.preprocess_image(file)
                
                genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), np.array(genders)
                    images, genders = [], []
                    
            if not is_training:
                break
    def return_df(self):
        return self.df
                
data_generator = FFFaceDataGenerator(df)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

batch_size = 32
valid_batch_size = 32
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

callbacks = [
    ModelCheckpoint("../model_checkpoint/gender2-cp-{epoch:04d}.hdf5", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=False, 
    save_weights_only=False, 
    mode='auto', 
    period=1)
]

model = load_model("../model_checkpoint/gender-cp-0084.hdf5") 

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=16,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


