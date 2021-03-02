from tensorflow.keras.applications import vgg16
from tensorflow import keras
from tensorflow.keras.applications import resnet_v2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Sequential
from tensorflow.python.framework import ops
import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow import keras
import sys
import cv2
from tensorflow.keras.models import load_model
import json
import dlib
from PIL import Image
tf.compat.v1.disable_eager_execution()

from keras.models import load_model
import os
import sys
import pandas as pd
from PIL import Image
import numpy as np
sys.path.insert(1, '../src')

#import preprocess_image
import importlib
#importlib.reload(preprocess_image)

import matplotlib.pyplot as plt


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

def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = path
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = resnet_v2.preprocess_input(x)
    #x = detect_face(img_path)
    return x

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)

def compile_saliency_function(model, activation_layer='conv2d_7'):
    input_img = model.input
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])
    layer_output = layer_dict[activation_layer].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def modify_backprop(model, name, model_path):
    g = tf.compat.v1.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
#         save_path = "../models/race_v6.hdf5"
        #save_path = model_path
        new_model = load_model(model_path)
    return new_model

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #if K.image_dim_ordering() == 'th':
    #    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(input_model, input_biased_model, image, category_index, biased_category_index, layer_name):
    model = input_model
    biased_model = input_biased_model
    #model = Sequential()
    #model.add(input_model)

    nb_classes = 7
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    biased_target_layer = lambda x: target_category_loss(x, biased_category_index, nb_classes)
    
    model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    
    biased_model.add(Lambda(biased_target_layer,
                     output_shape = target_category_loss_output_shape))

    loss = K.sum(model.layers[-1].output)
    #print("loss: ", loss)
    biased_loss = K.sum(biased_model.layers[-1].output)
    
    #conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output
    conv_output =  [l for l in model.layers if l.name == layer_name][0].output
    #print("conv_output: ", conv_output)
    biased_conv_output =  [l for l in biased_model.layers if l.name == layer_name][0].output
    
    
    grads = normalize(K.gradients(loss, conv_output)[0])
    biased_grads = normalize(K.gradients(biased_loss, biased_conv_output)[0])
    
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    biased_gradient_function = K.function([biased_model.layers[0].input], [biased_conv_output, biased_grads])
    

    output, grads_val = gradient_function([image])
    biased_output, biased_grads_val = biased_gradient_function([image])
    
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    biased_output, biased_grads_val = biased_output[0, :], biased_grads_val[0, :, :, :]
    

    weights = np.mean(grads_val, axis = (0, 1))
    biased_weights = np.mean(biased_grads_val, axis = (0, 1))
    
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)
    biased_cam = np.ones(biased_output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]
    for i, w in enumerate(biased_weights):
        biased_cam += w * biased_output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    biased_cam = cv2.resize(biased_cam, (224, 224))
    #print("cam1: ", cam)
    cam = np.maximum(cam, 0)
    biased_cam = np.maximum(biased_cam, 0)
    #print("cam2: ", cam)

    max_arr = np.concatenate([cam,biased_cam])
    max_norm = np.max(max_arr)

    heatmap = cam / max_norm
    biased_heatmap = biased_cam/ max_norm
    
    #print("heat: ", heatmap)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    biased_cam = cv2.applyColorMap(np.uint8(255*biased_heatmap), cv2.COLORMAP_JET)
    
    #heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    #biased_heatmap = cv2.applyColorMap(np.uint8(255*biased_heatmap), cv2.COLORMAP_JET)
    
    cam = np.float32(cam / 255) + np.float32(image)
    biased_cam = np.float32(biased_cam / 255) + np.float32(image)
    #heatmap = np.float32(cam / 255)
    
    cam = 255 * cam / np.max(cam)
    biased_cam = 255 * biased_cam / np.max(biased_cam)
    #heatmap = 255 * heatmap / np.max(heatmap)
    
    return np.uint8(cam), np.uint8(biased_cam), heatmap, biased_heatmap, image


#generate grad-cam and guided backprop with unbiased and biased models
def generate_grad_cam(PIL_img):
    with open("./config/parameters.json") as f:
        data = json.load(f)
    f.close()

    model_param = data["model_param"]
    lr, epochs, batch_size, mapping_path, save_path, log_path, biased_save_path = model_param.values()

#     with open(mapping_path) as f:
#         mapping = json.load(f)
#     f.close()
    
    print("load models")
    grad_params = data["grad_cam"]
    print("paths: ", save_path, biased_save_path)
    race_model = load_model(save_path)
    biased_race_model = load_model(biased_save_path)
    
    #grad_cam
    img_num = grad_params["img_idx"]
    sample_img_path = grad_params["image_path"] + str(img_num) + ".jpg"
    
    #preprocessed_input = load_image(sample_img_path)
    preprocessed_input = resnet_v2.preprocess_input(np.array(PIL_img)[None, :]) 
    
    print("start predictions")
    predictions = race_model.predict(preprocessed_input).squeeze()
    biased_preds = biased_race_model.predict(preprocessed_input).squeeze()

    predicted_class = np.argmax(predictions)
    biased_class = np.argmax(biased_preds)
    
    print("generate grad-cam")
    cam, biased_cam, heatmap, biased_heatmap, img = grad_cam(race_model, biased_race_model, preprocessed_input, predicted_class, biased_class, "conv2d_7")
    cv2.imwrite("GRADCAM_" + str(img_num) + ".jpg", cam)
    cv2.imwrite("BIASED_GRADCAM_" + str(img_num) + ".jpg", biased_cam)
    
    
    #guided_backprop
    print("generate guided backprop")
    register_gradient()
    guided_model = modify_backprop(race_model, 'GuidedBackProp', save_path)
    biased_guided_model = modify_backprop(biased_race_model, 'GuidedBackProp',biased_save_path)
    
    saliency_fn = compile_saliency_function(guided_model)
    biased_saliency_fn = compile_saliency_function(biased_guided_model)
    
    saliency = saliency_fn([preprocessed_input, 0])
    biased_saliency = biased_saliency_fn([preprocessed_input, 0])
    
    gradcam = saliency[0] * heatmap[..., np.newaxis]
    biased_gradcam = biased_saliency[0] * biased_heatmap[..., np.newaxis]
    
    cv2.imwrite("GUIDED_GRADCAM_" + str(img_num) + ".jpg", deprocess_image(gradcam))
    cv2.imwrite("BIASED_GUIDED_GRADCAM_" + str(img_num) + ".jpg", deprocess_image(biased_gradcam))


