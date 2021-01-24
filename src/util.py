import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense
from tensorflow.keras import Sequential
from IntegratedGradients import *
import json
from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

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
        np.random.seed(20)
        p = np.random.permutation(len(self.df))
        train_up_to = int(len(self.df) * self.TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]

        train_up_to = int(train_up_to * self.TRAIN_TEST_SPLIT)
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]
        
        # converts alias to id
        self.df['age_id'] = self.df['age'].map(lambda age: self.dataset_dict['age_alias'][age])

        print("Number of training data:{}".format(len(train_idx)))
        print("Number of validation data:{}".format(len(valid_idx)))
        print("Number of testing data:{}".format(len(test_idx)))
              
        return train_idx, valid_idx, test_idx
    
    def preprocess_image(self, img_path):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        im = Image.open(img_path)
        im = im.resize((self.IM_WIDTH, self.IM_HEIGHT))
        im = preprocess_input(np.array(im))
        
        #im = Image.open(img_path)
        #im = im.resize((self.IM_WIDTH, self.IM_HEIGHT))
        #im = np.array(im) / 255.0
        
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



def make_generator(train_label_path, train_image_path, valid_label_path, valid_image_path, resize, TRAIN_TEST_SPLIT):
    
    train_csv = pd.read_csv(train_label_path)
    valid_csv = pd.read_csv(valid_label_path)
    
    train_csv["file"] = train_csv["file"].apply(lambda x: os.path.join(train_image_path, x.split("/")[1]))
    valid_csv["file"] = valid_csv["file"].apply(lambda x: os.path.join(valid_image_path, x.split("/")[1]))
    
    combined = pd.concat([train_csv, valid_csv]).reset_index(drop = True)
    combined = combined.drop(["gender", "race", "service_test"], axis = 1)
    
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
    
    return DataGenerator(combined, TRAIN_TEST_SPLIT, dataset_dict, resize)
  

def preprocess_image(img_path, width, height):
        """
        Used to perform some minor preprocessing on the image before inputting into the network.
        """
        
        im = Image.open(img_path)
        im = im.resize((width, height))
        im = preprocess_input(np.array(im))
        
        #im = Image.open(img_path)
        #im = im.resize((width, height))
        #im = np.array(im) / 255.0
        
        return im
    
"""
Function to use the integrated_gradient to visualize the image
in: 
    model_param_path: saved model in .hdf5 format
    train_image_path: The image_path
    train_label_path: The label_path in .csv format
    save_path: path to save the image
    img_idx: The index of the image

out:
    original pictures
    annotation heatmaps
"""

def integrated_grad_pic(model_param_path, train_image_path, train_label_path, save_path, img_idx, size):
    model = keras.models.load_model(model_param_path)

    ig = integrated_gradients(model)

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
    
    img_name = "{}.jpg".format(img_idx)
    sample_path = os.path.join(train_image_path, img_name)
    sample_label_df = pd.read_csv(train_label_path)
    sample_label = sample_label_df[sample_label_df["file"].str.contains(img_name)]["age"].values[0]
    
    sample_image = preprocess_image(sample_path, size, size).reshape(1, size, size, 3)
    
    # Plot the true image.
    plt.figure(figsize = (5, 5))
    plt.imshow(sample_image.squeeze(), cmap="Greys")
    plt.xticks([],[])
    plt.yticks([],[])
    plt.title("Original image", fontsize=8)
    plt.savefig(os.path.join(save_path, "Original_") + str(img_idx)+".png")

    # Generate explanation with respect to each of 10 output channels.
    exs = []
    output_prob = model.predict(sample_image).squeeze()
    for i in range(1,10):
        exs.append(ig.explain(sample_image.squeeze(), outc=i-1))
    exs = np.array(exs)

    # Plot them
    th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))


    fig = plt.subplots(3,3,figsize=(15,15))
    for i in range(9):
        ex = exs[i]
        plt.subplot(3,3,i+1)
        plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title("heatmap for age range {} with probability {:.2f}".format(dataset_dict["age_id"][i],output_prob[i]), 
                  fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"integrated-viz_") + str(img_idx)+".png")
    plt.show()
    print("Ground Truth:", sample_label)
    print("Predicted:", dataset_dict["age_id"][np.argmax(output_prob)])

"""
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
"""
    
