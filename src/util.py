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
from tensorflow.keras.preprocessing.image import ImageDataGenerator


"""
Function to create generator from the csv file
input
    csv_path: path to the csv file
    image_path: path to the image directory
    target: the class of interest(age, gender, or race)
    size: the size of the image
    batch_size: the batch size
    preprocess_input: The preprocess function to apply based on different transfer learning model. Make sure to change
    the import statement above if wants to apply different transfer learning model
    mapping_path: a directionary objects indicating how each category is being mapped to the respective integer representation
    is_training: whether or not the generator is used as training

output
    a generator object ready to be trained
"""
def create_generator(csv_path, image_path, target, size, batch_size, mapping_path, preprocess_input, is_training):
    
    if is_training:
        horizontal_flip = False
        vertical_flip = False
    else:
        horizontal_flip = True
        vertical_flip = True
    
    df = pd.read_csv(csv_path)
    df["file"] = df["file"].apply(lambda x: os.path.join(image_path, x.split("/")[1]))
    
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        horizontal_flip = horizontal_flip, 
        vertical_flip = vertical_flip
        #rescale = 1 / 255
    )
    
    data_generator = imgdatagen.flow_from_dataframe(
        dataframe = df,
        directory = None,
        x_col = "file",
        y_col = target,
        target_size = (size, size),
        batch_size = batch_size,
        save_format = "jpg",
        shuffle = True
    )
    
    with open(mapping_path, "w") as f:
        json.dump(data_generator.class_indices, f)
    f.close()
    
    return data_generator
    

"""
Function to re-organize the dataset
input
    save_path: The new directory to save all the dataset
    train_csv_path, valid_csv_path, train_image_path, valid_image_path are self-explanatory
    target: the category to reorganized, such as age, gender, or raace
    
output will look similar to this(e.g. using gender):
    save_path
        train
            male
                images
                ...
            female
                images
                ...
        validation
            male
                images
                ...
            female
                images
                ...
"""
def create_dataset(save_path, train_csv_path, valid_csv_path, train_image_path, valid_image_path, target):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print("created dircetory: {}".format(save_path))
        
    else:
        print("dataset {} already exist!".format(save_path))
        return
        #shutil.rmtree(save_path)

        
    csv_path = [train_csv_path, valid_csv_path]
    image_path = [train_image_path, valid_image_path]
    names = ["train", "validation"]
    
    for i in range(len(csv_path)):
        df = pd.read_csv(csv_path[i])
        df["file"] = df["file"].apply(lambda x: os.path.join(image_path[i], x.split("/")[1]))
        grp_df = df.groupby(target)
        grps = grp_df.groups.keys()
        
        sub_dir = os.path.join(save_path, names[i])
        os.mkdir(sub_dir)
        print("created sub-directory: {}".format(sub_dir))
        
        for grp in grps:
            grp_dir = os.path.join(sub_dir, grp)
            os.mkdir(grp_dir)
            original_file_path = grp_df.get_group(grp)["file"]
            func = lambda x: os.path.join(grp_dir, x.split("/")[-1])
            new_file_path = original_file_path.apply(func).values
            original_file_path = original_file_path.values
            print("created category-directory: {}".format(grp_dir))
            
            for j in range(len(new_file_path)):
                img = PIL.Image.open(original_file_path[j])
                img.save(new_file_path[j])
    
    print("Finished!")
    
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
    

    
