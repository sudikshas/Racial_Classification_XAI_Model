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
import dlib
import io
from model_trans import *

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
    result_df.to_csv(os.path.join(save_path, "result_df.csv"), index = False)
    
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
function to create a bounding  box for the detected faces of an image
"""
def rect_to_bb(rect):
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

"""
Function to detect face from an image
input
    image_path: The path to the image. The image should include ONLY 1 face to align with the purpose of our web-app
    im_size: The size that the image should be resized
output
    an numpy array of an image that has been processed using the resnetv2.preprocess_input
"""
def detect_face(image_path, im_size = 224, default_max_size=800,size = 300, padding = 0.25):
    cnn_face_detector = dlib.cnn_face_detection_model_v1('./dlib_mod/mmod_human_face_detector.dat')
    sp = dlib.shape_predictor('./dlib_mod/shape_predictor_5_face_landmarks.dat')
    base = 2000  # largest width and height

    img = dlib.load_rgb_image(image_path)
    old_height, old_width, _ = img.shape
    old_height, old_width, _ = img.shape

    if old_width > old_height:
        new_width, new_height = default_max_size, int(default_max_size * old_height / old_width)
    else:
        new_width, new_height =  int(default_max_size * old_width / old_height), default_max_size
    img = dlib.resize_image(img, rows=new_height, cols=new_width)
    dets = cnn_face_detector(img, 1)
    num_faces = len(dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(image_path))
        return
    elif num_faces > 1:
        print("Multiple face in '{}'. A random face will be returned".format(image_path))
    faces = dlib.full_object_detections()
    for detection in dets:
        rect = detection.rect
        faces.append(sp(img, rect))
    image = dlib.get_face_chips(img, faces, size=size, padding = padding)[0]

    image = Image.fromarray(image, 'RGB')
    image = image.resize((im_size, im_size))

    #image = np.array(image) / 255.0
    #ori_img = np.array(image)
    processed_img = resnet_v2.preprocess_input(np.array(image))
    processed_img = processed_img[None,:]
    return processed_img

"""
function to get the prediction from the model
input
    img_path: The path to the image
    model_path: The path to the model
    mapping_path: The mapping
out
    The prediction made by the model
"""

"""
function to make a single prediction of an image
input
    img_path: The path to the image
    model_path: The path to the model
    mapping_path: The mapping between labels(in number) and categories
    result_df_path: The aggregate results
output
    out: The prediction
    pred_prob: The accuracy of making the out prediction
    aggregate_acc: The accuracy of the aggregate category

***NOTE: result_df_path can be found in(assuming race):
    "./visualization/race/stats/result_df.csv"
"""
def get_prediction(img_path, model_path, mapping_path, result_df_path):
    img_arr = detect_face(img_path)
    if img_arr.shape != (1, 224,224,3):
        print("Wrong input size")
        return
    else:
        model = keras.models.load_model(model_path)
        
        with open(mapping_path) as f:
            mapping = json.load(f)
        f.close()
        mapping = {val:key for key, val in mapping.items()}
        pred = model.predict(img_arr).squeeze()
        out = mapping[pred.argmax()]
        pred_prob = np.round(pred[pred.argmax()] * 100, 4)
        
        result_df = pd.read_csv(result_df_path)
        aggregate_acc = np.round(result_df[result_df["category"] == out]["accuracy"].values[0] * 100, 4)
    
    return out, pred_prob, aggregate_acc

"""
Function to use the integrated_gradient to visualize the image
in: 
    model_param_path: saved model in .hdf5 format
    image_path: The image_path
    label_path: The label_path in .csv format
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
        
def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

"""
functions to load the model with weights

in: weight_name of the checkpoint
out: the model loaded with weights
"""
def load_model_with_weights(weight_name):
    if "age" in weight_name:
        num_classes = 9
    elif "race" in weight_name:
        num_classes = 7
    else:
        num_classes = 2
    
    model = build_model(num_classes = num_classes)
    model.load_weights(weight_name)
    
    return model
    
"""
Function similar to integrated_grad_pic but just do it on one image

modification: the returned output is an image. This function no longer save the image into jpg.

in: 
    model_param_path: saved model in .hdf5 format
    mapping: The dictionary object of the mapping between labels and category
    target: The target(e.g. race, age, gender)
    image_path: The path to the image
out:
    a single image of object PIL.PngImage
"""
def integrated_grad_pic_single(model_param_path,mapping, target, image_path):
    
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
    
    
    processed_image = resnet_v2.preprocess_input(plt.imread(image_path)).reshape(-1, size, size, 3)

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
    fig = plt.gcf()
    plt.close()
    img = fig2img(fig)
    return img


"""
Another version of integrated_grad implementation that just shows the heatmap with the highest
predictive accuracy

NOTE: Before running this, Make sure you:
    1. Called Detect_face to crop the image only(WITHOUT USING Resnet Preprocessing)
    2. You should call resnet preprocessing unit INSIDE this function because
       the PIL.fromarray CANNOT take in float32 data type
       
   ALSO: Make sure you'd changed the model path and mapping path so that the function can run.

in: 
    PIL_img: a PIL_img object PIL.Image.Image
    
    target: the target(e.g. race, age, gender)
    
    lookup: The particular category to lookup. For instance, given target = race, lookup = None
            would display the heatmap with the highest probability. But if lookup = "white",
            the function would display the heatmap with "white" category even if the category
            does have have the highest probability.
   
out:
    a single image of object PIL.PngImagePlugin.PngImageFile
"""
def integrated_grad_PIL(PIL_img, target, lookup = None):
    if target == "race":
        model_path = "./models/race/race_v6.hdf5"
    elif target == "age":
        model_path = "./models/age/age_v1.hdf5"
    else:
        model_path = "./models/gender/gender_v1.hdf5"
        
    model = keras.models.load_model(model_path)
    ig = integrated_gradients(model)

    mapping = os.path.join("./mapping", target + ".json")
    with open(mapping) as f:
        mapping_dict = json.load(f)
    f.close()

    mapping_dict = {key.lower():val for key, val in mapping_dict.items()}
    mapping_dict_rev = {val:key for key, val in mapping_dict.items()}
    
    ############################THIS LINE IS IMPORTANT!!!!#################################
    PIL_img = resnet_v2.preprocess_input(np.array(PIL_img)[None,:]) ##IMPORTANT!!!
    output_prob = model.predict(PIL_img).squeeze()
    pred_idx = output_prob.argmax()
    
    if lookup == None:
        pass
    else:
        lookup = lookup.lower()
        pred_idx = mapping_dict[lookup]

    ex = ig.explain(PIL_img.squeeze(), outc=pred_idx)

    th = max(np.abs(np.min(ex)), np.abs(np.max(ex)))

    plt.figure(figsize = (6, 6))
    plt.imshow(ex[:,:,0], cmap="seismic", vmin=-1*th, vmax=th)
    plt.title("heatmap for {} {} with probability {:.2f}".format(target, mapping_dict_rev[pred_idx],
                                                                 output_prob[pred_idx]), fontsize=12)
    
    fig = plt.gcf()
    im = fig2img(fig)

    return im
             