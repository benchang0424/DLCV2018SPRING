import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from models import *
from read import *
import argparse
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))

BATCH_SIZE = 10


def train(mode,TRAIN_DIR,VAL_DIR):
    #TRAIN_DIR = "hw3-train-validation/train/"
    #VAL_DIR = "hw3-train-validation/validation/"

    print("loading......")
    x_train = read_images(TRAIN_DIR)
    x_val = read_images(VAL_DIR)
    y_train = read_masks(TRAIN_DIR)
    y_val = read_masks(VAL_DIR)

    x_train = x_train.astype(np.float32) / 255.0
    x_val = x_val.astype(np.float32) / 255.0
    y_train = to_categorical(y_train, 7)
    y_val = to_categorical(y_val, 7)

    print("loading data done!")
    model = FCN_Vgg16(input_shape=(512, 512, 3), mode=mode)
    print("loading models done!")

    #opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adamax', metrics = ['accuracy'])
    model.summary()
    save_model_path = os.path.join('save_models',mode)
    checkpoint = ModelCheckpoint(save_model_path+'/epoch_{epoch:02d}.h5', monitor='val_acc',save_best_only=True, verbose=1, mode='max')
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')
    model.fit(x_train, y_train, validation_data=(x_val, y_val)
                ,callbacks = [early_stopping, checkpoint]
                ,epochs=80, batch_size=BATCH_SIZE, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-mo', '--mode', help='training mode', type=str)
    parser.add_argument('-t', '--train', help='training images directory', type=str)
    parser.add_argument('-v', '--val', help='validation images directory', type=str)
    args = parser.parse_args()
    mode = args.mode
    TRAIN_DIR = args.train
    VAL_DIR = args.val

    train(mode,TRAIN_DIR,VAL_DIR)

