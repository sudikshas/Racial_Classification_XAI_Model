import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Activation, Dropout, Lambda, Dense
from tensorflow.keras import Sequential
from IntegratedGradients import *
import json
from tensorflow import keras
#from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.applications import resnet_v2
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


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
        rotation_range = 30
        horizontal_flip = True
        vertical_flip = True
        shuffle = True
    else:
        rotation_range = 0
        horizontal_flip = False
        vertical_flip = False
        shuffle = False
    
    df = pd.read_csv(csv_path)
    df["file"] = df["file"].apply(lambda x: os.path.join(image_path, x.split("/")[1]))
    
    imgdatagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = rotation_range,
        horizontal_flip = horizontal_flip, 
        vertical_flip = vertical_flip,
        #rescale = 1.0 / 255
    )
    
    data_generator = imgdatagen.flow_from_dataframe(
        dataframe = df,
        directory = None,
        x_col = "file",
        y_col = target,
        target_size = (size, size),
        batch_size = batch_size,
        save_format = "jpg",
        shuffle = shuffle
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
function to visualize the training progress
input
    log_path: The csv file that logged the training progress
    target: the name of the class (e.g. age, race, gender)
output
    the accuray and loss curve for both the training and validation
"""
def generate_curves(log_path, save_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
       
    df = pd.read_csv(log_path)
    path_to_viz = save_path
    acc_name = os.path.join(path_to_viz, "acc_curve")
    loss_name = os.path.join(path_to_viz, "loss_curve")
    
    ax = plt.gca()
    plt.plot(df["accuracy"])
    plt.plot(df["val_accuracy"])
    plt.title("Training Accuracy vs. Validation Accuracy")
    plt.xlabel("epochs")
    plt.ylabel("Accuracy")
    ax.legend(['Train','Validation'],loc='lower right')
    plt.savefig(acc_name)
    plt.close()
    
    ax = plt.gca()
    plt.plot(df["loss"])
    plt.plot(df["val_loss"])
    plt.title("Training loss vs. Validation loss")
    plt.xlabel("epochs")
    plt.ylabel("Loss")
    ax.legend(['Train','Validation'],loc='upper right')
    plt.savefig(loss_name)
    

"""
Function to evaluate the model by calculating category-specific statistics
input
    model: a loaded model
    generator: a generator that contains data
    label_df: the csv file that contains the information of the data
    target: the name of the class (e.g. age, race, gender)
    target_map: The mapping of the class
    save_path: where the plot should be saved
output
    The class-specific barplot for precision, recall, f1-score, accuracy, and support
"""
def create_stats(model, generator, target, label_path, mapping_path, save_path):
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    label_df = pd.read_csv(label_path)
    with open(mapping_path) as f:
        target_map = json.load(f)
    f.close()
    
    pred = model.predict(generator).argmax(axis = 1)
    ground_truth = label_df[target].replace(target_map).values
    cr = classification_report(ground_truth, pred, target_names = target_map.keys())
    
    with open(os.path.join(save_path, "class_report.txt"), "w") as f:
        f.write(cr)
    f.close()
    
    cr = classification_report(ground_truth, pred, target_names = target_map.keys(), output_dict = True)
    
    result_df = pd.DataFrame(cr).T.iloc[:len(target_map), :]
    result_df = result_df.reset_index().rename(columns= {"index": "category"})

    cm = confusion_matrix(ground_truth, pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    acc = cm.diagonal()
    result_df["accuracy"] = acc
    
    stat_names = ["precision", "recall", "f1-score", "accuracy", "support"]
    
    for name in stat_names:
        save_dir = os.path.join(save_path, name + "_barplot")
        plt.figure(figsize = (12,8))
        sns.barplot(x = "category", y= name, data= result_df,linewidth=2.5, 
                    facecolor=(1, 1, 1, 0), edgecolor="0")
        plt.title("{} across {}".format(name, target), fontsize = 20)
        plt.xlabel(target, fontsize = 16)
        plt.ylabel(name, fontsize= 16)
        plt.savefig(save_dir)  
        
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

def integrated_grad_pic(model_param_path, image_path, label_path, save_path, target, mapping, size, img_idx_lst):
    
    model = keras.models.load_model(model_param_path)
    ig = integrated_gradients(model)
    
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()
    
    mapping_dict = {val:key for key, val in mapping_dict.items()}
    
    max_iter_range = len(mapping_dict)
    if target == "age" or target == "race":
        subplot_row = 3
        subplot_col = 3
    else:
        subplot_row = 1
        subplot_col = 2
    
    for img_idx in img_idx_lst:
        img_name = "{}.jpg".format(img_idx)
        sample_path = os.path.join(image_path, img_name)
        sample_label_df = pd.read_csv(label_path)
        sample_label = sample_label_df[sample_label_df["file"].str.contains(img_name)][target].values[0]

        sample_image = Image.open(sample_path)
        sample_image.save(os.path.join(save_path, "Original_") + str(img_idx)+".png")

        processed_image = resnet_v2.preprocess_input(plt.imread(sample_path)).reshape(-1, size, size, 3)

        exs = []
        output_prob = model.predict(processed_image).squeeze()
        for i in range(1, max_iter_range + 1):
            exs.append(ig.explain(processed_image.squeeze(), outc=i-1))
        exs = np.array(exs)

        # Plot them
        th = max(np.abs(np.min(exs)), np.abs(np.max(exs)))

        fig = plt.subplots(subplot_row, subplot_col,figsize=(15,15))
        for i in range(max_iter_range):
            ex = exs[i]
            plt.subplot(subplot_row,subplot_col,i+1)
            plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
            plt.xticks([],[])
            plt.yticks([],[])
            plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict[i],output_prob[i]), 
                      fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path,"integrated-viz_") + str(img_idx)+".png")
        plt.close()
        print("Ground Truth for {}:".format(img_idx), sample_label)
        print("Predicted for {}:".format(img_idx), mapping_dict[np.argmax(output_prob)])
             

    
