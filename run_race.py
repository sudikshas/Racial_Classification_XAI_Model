import numpy as np 
import pandas as pd
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset_folder_name = '../../fairface_data'

TRAIN_TEST_SPLIT = 0.7
IM_WIDTH = IM_HEIGHT = 198

dataset_dict = {
    'race_id': {
        0: 'White', 
        1: 'Black', 
        2: 'East Asian', 
        3: 'Indian', 
        4: 'Middle Eastern',
        5: 'Latino_Hispanic',
        6: 'Southeast Asian'
    },
    'gender_id': {
        0: 'Male',
        1: 'Female'
    }
}

dataset_dict['gender_alias'] = dict((g, i) for i, g in dataset_dict['gender_id'].items())
dataset_dict['race_alias'] = dict((g, i) for i, g in dataset_dict['race_id'].items())


dataset_folder_name = '../../fairface_data' 
dataset_csv = '/fairface_label_train.csv'
dataset = pd.read_csv(dataset_folder_name + dataset_csv)

dataset["file"] = dataset["file"].apply(
    lambda x:"../../fairface_data/fairface_pad025/"+x)

#Data Generator
from keras.utils import to_categorical
from PIL import Image

class FaceDataGenerator():
    """
    Data generator for the UTKFace dataset. This class should be used when training our Keras multi-output model.
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
        #self.df['gender_id'] = self.df['gender'].map(lambda gender: dataset_dict['gender_alias'][gender])
        self.df['race_id'] = self.df['race'].map(lambda race: dataset_dict['race_alias'][race])

        #self.max_age = self.df['age'].max()
        
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
        #images, ages, races, genders = [], [], [], []
        images, races = [], []
        while True:
            for idx in image_idx:
                person = self.df.iloc[idx]
                
                #age = person['age']
                race = person['race_id']
                #gender = person['gender_id']
                file = person['file']
                
                im = self.preprocess_image(file)
                #print("im_size: ", im.shape)
                
                #ages.append(age / self.max_age)
                #ages.append(0)
                races.append(to_categorical(race, len(dataset_dict['race_id'])))
                #genders.append(to_categorical(gender, len(dataset_dict['gender_id'])))
                images.append(im)
                
                # yielding condition
                if len(images) >= batch_size:
                    #yield np.array(images), [np.array(ages), np.array(races), np.array(genders)]
                    yield np.array(images), np.array(races)
                    #images, ages, races, genders = [], [], [], []
                    images, races = [], []
                    
            if not is_training:
                break
                
data_generator = FaceDataGenerator(dataset)
train_idx, valid_idx, test_idx = data_generator.generate_split_indexes()

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

class MultiOutputModel():
    """
    Used to generate our multi-output model. This CNN contains three branches, one for age, other for 
    sex and another for race. Each branch contains a sequence of Convolutional Layers that is defined
    on the make_default_hidden_layers method.
    """
    def make_default_hidden_layers(self, inputs):
        """
        Used to generate a default set of hidden layers. The structure used in this network is defined as:
        
        Conv2D -> BatchNormalization -> Pooling -> Dropout
        """
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(3, 3))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        x = Conv2D(32, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)

        return x

    def build_race_branch(self, inputs, num_races):
        """
        Used to build the race branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_races)(x)
        x = Activation("softmax", name="race_output")(x)

        return x

    def build_gender_branch(self, inputs, num_genders=2):
        """
        Used to build the gender branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.
        """
        x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)

        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(num_genders)(x)
        x = Activation("sigmoid", name="gender_output")(x)

        return x

    def build_age_branch(self, inputs):   
        """
        Used to build the age branch of our face recognition network.
        This branch is composed of three Conv -> BN -> Pool -> Dropout blocks, 
        followed by the Dense output layer.

        """
        x = self.make_default_hidden_layers(inputs)

        x = Flatten()(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        x = Activation("linear", name="age_output")(x)

        return x

    def assemble_full_model(self, width, height, num_races):
        """
        Used to assemble our multi-output model CNN.
        """
        input_shape = (height, width, 3)

        inputs = Input(shape=input_shape)

        #age_branch = self.build_age_branch(inputs)
        race_branch = self.build_race_branch(inputs, num_races)
        #print(race_branch.shape)
        #gender_branch = self.build_gender_branch(inputs)

#         model = Model(inputs=inputs,
#                      outputs = [age_branch, race_branch, gender_branch],
#                      name="face_net")
        model = Model(inputs=inputs,
                     outputs = race_branch,
                     name="face_net")

        return model
    
model = MultiOutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT, num_races=len(dataset_dict['race_alias']))


from keras.optimizers import Adam

init_lr = 1e-4
epochs = 1

opt = Adam(lr=init_lr, decay=init_lr / epochs)

model.compile(optimizer=opt, 
              loss={
                  'race_output': 'categorical_crossentropy'},
              loss_weights={
                  'race_output': 1.5},
              metrics={
                  'race_output': 'accuracy'})

from keras.callbacks import ModelCheckpoint
batch_size = 32
valid_batch_size = 32
# batch_size = 128
# valid_batch_size = 128
train_gen = data_generator.generate_images(train_idx, is_training=True, batch_size=batch_size)
valid_gen = data_generator.generate_images(valid_idx, is_training=True, batch_size=valid_batch_size)

# callbacks = [
#     ModelCheckpoint(filepath="./model_checkpoint/race-cp-{epoch:04d}.h5",
#                     save_weights_only = False,
#                     monitor='val_loss',
#                     verbose = 1,
#                     save_best_only=False,
#                     mode = "min")
# ]

callbacks = [
    ModelCheckpoint("../model_checkpoint/gender-cp-{epoch:04d}.hdf5", 
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False, 
    mode='auto', 
    period=1)
]

from keras.callbacks import CSVLogger

csv_logger = CSVLogger('training.log', separator=',', append=False)

history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=epochs,
                    callbacks=[callbacks, csv_logger],
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)

model.save('racial_classification_model.hdf5')




