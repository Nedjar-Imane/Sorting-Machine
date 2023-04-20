import matplotlib as plt
import matplotlib.pyplot as plt
%matplotlib inline
from tensorflow import keras 
from tensorflow.keras import layers
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D,Dropout,Flatten
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.applications.mobilenet import MobileNet,preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys
import pandas as pd
from tensorflow.keras import backend as K
import argparse
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
  
early_stop = EarlyStopping(
     monitor='loss',
     min_delta=0.001,
     patience=20,
     mode='min',
     verbose=1
)
checkpoint = ModelCheckpoint(
     '/mobile/model_best_weights.h5', 
     monitor='loss', 
     verbose=1, 
     save_best_only=True, 
     mode='min', 
     period=1
)
def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
    base_model: keras model excluding top
    nb_classes: # of classes
    Returns:
    new keras model with last layer
    """
    FC_SIZE=512
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)
    x=Dropout(0.5)(x)
    x = Dense(224, activation='relu')(x)
    x=Dropout(0.5)(x)
    predictions =Dense(1, activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    return model

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    
    for layer in base_model.layers[:80]:
      layer.trainable = False
    for layer in base_model.layers[80:]:
        layer.trainable = True  #lr=0.0001
    
def mobileNet(NB_EPOCHS,BAT_SIZE,train_path,validation_path,test_path,Class_mode,nb_classes,optimizer):

  train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input )
  test_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input)
  nb_classes= int(nb_classes) 
  NB_EPOCHS=NB_EPOCHS
  BAT_SIZE=BAT_SIZE
  train_path=train_path
  validation_path=validation_path
  test_path=test_path
  Class_mode=Class_mode
  
  train_generator = train_datagen.flow_from_directory(
  directory= train_path,
  target_size=(224, 224),
  batch_size=BAT_SIZE,
  class_mode = Class_mode)
  
  test_generator = test_datagen.flow_from_directory(
  directory=test_path,
  target_size=(224, 224),
  batch_size=BAT_SIZE,
  class_mode = Class_mode,
  shuffle=False)
  
  base_model =MobileNet(input_shape=(224, 224, 3),include_top=False, weights='imagenet',classes=2)
  model = add_new_last_layer(base_model, nb_classes)
  # transfer learning
  setup_to_transfer_learn(model, base_model)
  model.compile(loss="binary_crossentropy", optimizer=optimizer,
                       metrics=["accuracy"])

  
  hist_tf=model.fit_generator(
  train_generator,
  epochs=NB_EPOCHS,
  validation_data=test_generator,
  callbacks = [early_stop,checkpoint]
      )
     
  model_json = model.to_json()
  with open("/mobile/model.json", "w") as json_file:
    json_file.write(model_json)
  # load json and create model
  json_file = open('/mobile/model.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  
  loaded_model = model_from_json(loaded_model_json)
  # load weights into new model
  loaded_model.load_weights("/mobile/model_best_weights.h5")
  print("Loaded model from disk")
  loaded_model.compile(loss="binary_crossentropy", optimizer=optimizer,
                       metrics=["accuracy"])

 
  # evaluate loaded model on test data
  scoreSeg =loaded_model.evaluate_generator(generator=test_generator)
  print("Accuracy = ",scoreSeg[1])
  
  
  labels = (train_generator.class_indices)
    
  return hist_tf






BATCH_SIZE = 32
EPOCHS = 20
INIT_LR = 1e-4
MAX_LR = 1e-1
train_path='/combined-dataset/train'
validation_path='/combined-dataset/test'
test_path='/combined-dataset/test'
###########################
steps_per_epoch = 1590 // BATCH_SIZE
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size=2 * steps_per_epoch
)
optimizer = tf.keras.optimizers.SGD(clr)
#############################""

Class_mode='binary'
nb_classes=2
hist_tf=mobileNet(EPOCHS,BATCH_SIZE,train_path,validation_path,test_path,Class_mode,nb_classes,optimizer)
print(hist_tf.history)
epoch_list = list(range(1, len(hist_tf.history['accuracy']) + 1))
loss_epoch_list = list(range(1, len(hist_tf.history['loss']) + 1))
plt.plot(epoch_list, hist_tf.history['accuracy'], epoch_list, hist_tf.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.show()
plt.plot(loss_epoch_list, hist_tf.history['loss'], loss_epoch_list, hist_tf.history['val_loss'])
plt.legend(('Training Loss', 'Validation Loss'))
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.ylim([0, 1])
plt.show()
