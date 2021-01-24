import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

##THIS IS IMPORTANT!
tf.compat.v1.disable_eager_execution()

def training(model, data_generator, train_batch_size, valid_batch_size, lr, epochs, save_path, log_path):


    train_idx, valid_idx,test_idx = data_generator.generate_split_indexes()
    train_gen = data_generator.generate_images(train_idx, is_training = True, batch_size = train_batch_size)
    valid_gen = data_generator.generate_images(train_idx, is_training = True, batch_size = valid_batch_size)

    init_lr = lr
    epochs = epochs

    optimizer = Adam(lr=init_lr) #, decay=init_lr / epochs

    model.compile(optimizer=optimizer, 
                  loss= 'categorical_crossentropy', 
                  metrics= ["accuracy"])

    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))


    callbacks = [
        ModelCheckpoint(
                        filepath = save_path,
                        save_weights_only = False, #previously set True
                        save_best_only = True,
                        monitor='val_loss',
                        mode = "min"),

        EarlyStopping(monitor = "val_loss",
                      patience = 10,
                      mode = "min"),
        
        CSVLogger(log_path,
                  separator = ",",
                  append = False),
        
        ReduceLROnPlateau(monitor='val_loss',
                          factor=0.5,
                          patience=5,
                          mode = "min")
        ]

    print("curdir:", os.path.abspath(os.curdir))
    print("save_path:", save_path)
    history = model.fit_generator(train_gen,
                    steps_per_epoch=len(train_idx)//train_batch_size,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)
    
    return history
    
