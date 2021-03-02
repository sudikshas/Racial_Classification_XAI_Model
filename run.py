import sys
sys.path.append("./src")
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from model_trans import *
from util import *
from test_func import *
from training import *
import json
from tensorflow.keras.applications import resnet_v2
import pandas as pd

if __name__ == '__main__':
    
    targets = sys.argv[1:]
   
    with open("./config/parameters.json") as param:
        data = json.load(param)
    param.close()
    
    model_param = data["model_param"]
    data_info = data["load_data"]
    generate_stats = data["generate_stats"]
    integrated_grad = data["integrated_grad"]
    run_your_img = data["run_your_img"]

    #Train model
    if "train_model" in targets:

        lr, epochs, batch_size, mapping_path, save_path, log_path = model_param.values()

        train_label_path, train_image_path, valid_label_path, valid_image_path, target, size = data_info.values()

        num_classes = pd.read_csv(valid_label_path)[target].nunique()


        train_gen = create_generator(train_label_path,
                                     train_image_path,
                                     target,
                                     size,
                                     batch_size,
                                     mapping_path,
                                     resnet_v2.preprocess_input, 
                                     is_training = True)

        valid_gen = create_generator(valid_label_path,
                                     valid_image_path,
                                     target,
                                     size,
                                     batch_size,
                                     mapping_path,
                                     resnet_v2.preprocess_input, 
                                     is_training = False)

        #print("number of training data:", len(train_gen) * batch_size)
        #print("number of validation data:", len(valid_gen) * batch_size)

        model = build_model(num_classes = num_classes)

        print(model.summary())

        training(model, train_gen, valid_gen, lr, epochs, save_path, log_path)
    
    #generate statistics
    if "generate_plots" in targets:
        log_path, label_path, image_path, target, save_path, model_path = generate_stats.values()
        model = keras.models.load_model(model_path)
        mapping_path = os.path.join("./mapping", target + ".json")

        generator = create_generator(label_path,
                                         image_path,
                                         target,
                                         224,
                                         128,
                                         mapping_path,
                                         resnet_v2.preprocess_input, 
                                         is_training = False)


        generate_curves(log_path, save_path)
        create_stats(model, generator, target, label_path, mapping_path, save_path)
    
    #integrated-gradient
    if "integrate_grad" in targets:
        integrated_grad_pic(**integrated_grad)
    
    #test your own image
    if "run_pic" in targets:
        image_path, target, mapping, to_save = run_your_img.values()
        to_save = bool(to_save)
        face_img = detect_face(image_path, to_save = to_save)
        integrated_grad_PIL(face_img, target, to_save = to_save)
        


    
