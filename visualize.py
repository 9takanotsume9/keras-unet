import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
from distutils.util import strtobool
from tqdm import tqdm

import tensorflow as tf
import keras.backend as K

from ncc.utils.image import random_colors, apply_mask
from ncc.video.utils import FPS

from models import UNET


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('gpu', type=str, default=3)
    parser.add_argument('function', help='Function name in this script file.')
    parser.add_argument('--conf_thresh', help='Threshold of confidence.', default=0.1, type=float)
    parser.add_argument('--model_path', help='Path to model file (ie. /weights/sample.h5).')
    parser.add_argument('--num', help='Number of predicted images.')
    parser.add_argument('--root', help='Path to dataset directory.')
    parser.add_argument('--shuffle', type=strtobool, default=False)
    parser.add_argument('--start_frame', help='Frame id of starting for target video.', type=int, default=10000)
    parser.add_argument('--video_path', help='Path to video of target.')
    
    return parser.parse_args()


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    return tf.Session(config=config)


def read_image(path, input_shape, as_color=True):
    image = cv2.imread(str(path), as_color)
    image = cv2.resize(image, (input_shape[0], input_shape[1]))
    if not as_color:
        image = image.astype('bool').astype('float32')
        # image = np.expand_dims(image, axis=-1)

    return image


def build_model(weights):
    assert Path(weights).exists(), 'Model path is invalid.'
    model = UNET()
    model.load_weights(weights)

    return model


def predict(model, image, channel, conf_thresh, rgb=True):
    colors = random_colors(channel, scale=False)
    blend = image.copy()
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = image.astype('float32')/255
    pred = model.predict(image[np.newaxis, :, :, :])[0]
    pred[pred > conf_thresh] = 1
    pred = pred.astype('uint8')
    for i in range(channel):
        blend = apply_mask(blend, pred[:,:,i], colors[i])
    if rgb:
        blend = cv2.cvtColor(blend, cv2.COLOR_BGR2RGB)

    return blend


def predict_image(model_path, dataset_dir, num, conf_thresh, shuffle=False):
    model = build_model(model_path)
    (height, width), channel = model.input_shape[1:3], model.output_shape[3]
    images_dir = Path(dataset_dir)/'images'
    labels_dir = Path(dataset_dir)/'labels'

    images_path = sorted([Path(images_dir)/x for x in images_dir.iterdir()])
    if shuffle:
        random.shuffle(images_path)

    for image_path in images_path[:int(num)]:
        image = read_image(image_path, (width, height), as_color=True)
        blend = predict(model, image, channel, conf_thresh, rgb=True)
        label_path = Path(labels_dir)/image_path.name
        label = read_image(label_path, (blend.shape[1], blend.shape[0]), as_color=False)

        plt.figure(figsize=(16,16))
        plt.subplot(1,2,1)
        plt.imshow(blend)
        plt.subplot(1,2,2)
        plt.imshow(label)
        plt.show()


def predict_video(model_path, video_path, conf_thresh, start_frame):
    model = build_model(model_path)
    (height, width), channel = model.input_shape[1:3], model.output_shape[3]
    cap = cv2.VideoCapture(video_path)
    cap.set(1, start_frame)
    fps = FPS()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image = cv2.resize(frame, (width, height))
        blend = predict(model, image, channel, conf_thresh, rgb=False)
        fps.calculate(blend)
        cv2.imshow('Predict', blend)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.function == 'predict_image':
        predict_image(args.model_path, args.root, args.num, args.conf_thresh, args.shuffle)
    elif args.function == 'predict_video':
        predict_video(args.model_path, args.video_path, args.conf_thresh, args.start_frame)
