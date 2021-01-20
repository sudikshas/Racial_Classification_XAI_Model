import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense
from tensorflow.keras import Sequential

class DataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
    """
    def __init__(self, df, TRAIN_TEST_SPLIT, dataset_dict, resize):
        self.df = df
        self.TRAIN_TEST_SPLIT = TRAIN_TEST_SPLIT
        self.dataset_dict = dataset_dict
        self.IM_WIDTH = resize
        self.IM_HEIGHT = resize
        
    def generate_split_indexes(self):
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * self.TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * self.TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['age_id'] = self.df['age'].map(lambda age: self.dataset_dict['age_alias'][age])

        
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((self.IM_WIDTH, self.IM_HEIGHT))
        im = np.array(im) / 255.0
        
        return im
        
    def generate_images(self, image_idx, is_training, batch_size=16):
        """
        Used to generate a batch with images when training/testing/validating our Keras model.
        """
        
        # arrays to store our batched data
        images, ages = [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                age = person['age_id']
                file = person["file"]
                
                im = self.preprocess_image(file)
                
                ages.append(to_categorical(age, len(self.dataset_dict["age_id"])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    yield np.array(images), [np.array(ages)]
                    images, ages = [], []
                    
            if not is_training:
                break

def make_generator(train_label_path, train_image_path, resize, TRAIN_TEST_SPLIT):
  
    train_csv_df = pd.read_csv(train_label_path)

    img_name = os.listdir(train_image_path)
    img_name = sorted(img_name, key = lambda x: int(x.split(".")[0]))
    img_name = list(map(lambda x: os.path.join(train_image_path, x), img_name))

    train_csv_df["file"] = img_name
    working = train_csv_df.drop(["gender", "race", "service_test"], axis = 1)

    dataset_dict = {
        'age_id': {
            0: '0-2', 
            1: '3-9', 
            2: '10-19', 
            3: '20-29', 
            4: '30-39',
            5: '40-49',
            6: '50-59',
            7: '60-69',
            8: "more than 70"
        }
    }

    dataset_dict['age_alias'] = dict((g, i) for i, g in dataset_dict['age_id'].items())
    
    return DataGenerator(working, TRAIN_TEST_SPLIT, dataset_dict, resize)
    