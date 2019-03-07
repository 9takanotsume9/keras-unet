import argparse
import cv2
import numpy as np
import os
from pathlib import Path
import random
from tqdm import tqdm

import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from ncc.history.history import save_history

from models import UNET, dice_coef, unet_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Training script for neural network.')
    parser.add_argument('gpu', type=str, default=3)
    parser.add_argument('root', help='Path to the directory that contains images and labels.')
    parser.add_argument('checkpoint', help='File path of saved weight.')
    parser.add_argument('history', help='File path of training history.')
    parser.add_argument('--batch_size', help='Size of the batches.', type=int, default=1)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=200)
    
    return parser.parse_args()


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def read_image(path, input_shape, as_color=True):
    image = cv2.imread(str(path), as_color)
    image = cv2.resize(image, (input_shape[1], input_shape[0]))
    if not as_color:
        image = image.astype('bool').astype('float32')
        image = np.expand_dims(image, axis=-1)

    return image


def my_generator(dataset_dir, input_shape, batch_size):
    inputs, targets = [], []
    imgs_path = sorted((Path(dataset_dir)/'images').iterdir())
    labs_path = sorted((Path(dataset_dir)/'labels').iterdir())
    while True:
        for img_path, lab_path in zip(imgs_path, labs_path):
            inputs.append(read_image(img_path, input_shape, as_color=True))
            targets.append(read_image(lab_path, input_shape, as_color=False))
            if len(inputs) == batch_size:
                inputs = np.array(inputs).astype('float32')/255.
                targets = np.array(targets).astype('float32')

                yield (inputs, targets)
                inputs, targets = [], []


def train(dataset_dir, checkpoint_path, history_path, batch_size, epochs):
    INPUT_SHAPE = (360,640,3)
    model = UNET()
    model.compile(
        loss=unet_loss,
        optimizer=SGD(lr=0.02, momentum=0.9),
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='loss', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')
    ]

    history = model.fit_generator(
        my_generator(dataset_dir, INPUT_SHAPE, batch_size),
        steps_per_epoch=len(list((Path(dataset_dir)/'images').iterdir())),
        epochs=epochs,
        # validation_data=my_generator(Path(dataset_dir)/'val', INPUT_SHAPE, batch_size),
        # validation_steps=len(list((Path(dataset_dir)/'val'/'images').iterdir())),
        callbacks=callbacks
    )

    save_history(history, history_path)


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    K.tensorflow_backend.set_session(get_session())
    train(args.root, args.checkpoint, args.history, args.batch_size, args.epochs)