import sys
sys.path.append("./src")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model_trans import *
from util import *
from training import *
import json
#from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import mobilenet_v2
from tensorflow.keras.applications import resnet_v2
import pandas as pd

if __name__ == '__main__':
    ###load parameters
    with open("./config/age_parameters.json") as param:
        data = json.load(param)
        model_param = data["model_param"]
        data_info = data["load_data"]

    param.close()

    lr, epochs, batch_size, mapping_path, save_path, log_path = model_param.values()
    
    train_label_path, train_image_path, valid_label_path, valid_image_path, target, size = data_info.values()
    
    num_classes = pd.read_csv(valid_label_path)[target].nunique()

    
    train_gen = create_generator(train_label_path,
                                 train_image_path,
                                 target,
                                 size,
                                 batch_size,
                                 mapping_path,
                                 resnet_v2.preprocess_input, ##change this
                                 is_training = True)
    
    valid_gen = create_generator(valid_label_path,
                                 valid_image_path,
                                 target,
                                 size,
                                 batch_size,
                                 mapping_path,
                                 resnet_v2.preprocess_input, ##change this
                                 is_training = False)
    
    model = build_model(num_classes = num_classes)
    
    print(model.summary())
    
    ###training
    
    training(model, train_gen, valid_gen, lr, epochs, save_path, log_path)


    