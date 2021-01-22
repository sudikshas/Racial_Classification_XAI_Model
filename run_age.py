import sys
sys.path.append("./src")
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from model import *
from util import *
from training import *
import json


if __name__ == '__main__':
    ###load parameters
    with open("./config/age_parameters.json") as param:
        data = json.load(param)
        model_param = data["model_param"]
        data_info = data["load_data"]

    param.close()

    lr = model_param["lr"]
    epochs = model_param["epochs"]
    train_batch_size = model_param["train_batch_size"]
    valid_batch_size = model_param["valid_batch_size"]
    save_path = model_param["save_path"]
    log_path = model_param["log_path"]
    IM_WIDTH = IM_HEIGHT = data_info["resize"]

    ###preparing data and model
    data_generator = make_generator(**data_info)
    model = OutputModel().assemble_full_model(IM_WIDTH, IM_HEIGHT)
    
    ###training
    
    training(model, data_generator, train_batch_size, valid_batch_size, lr, epochs, save_path, log_path)


    
