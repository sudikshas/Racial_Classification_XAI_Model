import json
import pandas as pd
import os
from tensorflow import keras
import sys
sys.path.append("./src")
from util import *
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

if __name__ == '__main__':
   
    with open("./config/parameters.json") as param:
        all_info = json.load(param)
        generate_stats = all_info["generate_stats"]
        
    param.close()
    
    log_path, label_path, image_path, target, save_path, model_path = generate_stats.values()
    model = keras.models.load_model(model_path)
    mapping_path = os.path.join("./mapping", target + ".json")
 
    generator = create_generator(label_path,
                                     image_path,
                                     target,
                                     224,
                                     128,
                                     mapping_path,
                                     resnet_v2.preprocess_input, ##change this
                                     is_training = False)
    
    
    generate_curves(log_path, save_path)
    create_stats(model, generator, target, label_path, mapping_path, save_path)
    


    
