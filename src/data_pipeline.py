import os
import glob
import random
import pathlib
import tensorflow as tf
import numpy as np
from src.config import *
from src.config import CLASS_NAMES

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# CLASS_NAMES = np.array([
#     item.name for item in pathlib.Path(TRAIN_DIR).glob('*') if item.name != "LICENSE.txt"
# ])

def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    return parts[-2] == CLASS_NAMES

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMAGE_SHAPE)
    image = image / 255.0
    return image

def load_and_preprocess_data(path):
    image = load_and_preprocess_image(path)
    label = get_label(path)
    return image, label

def build_dataset(file_paths,  shuffle_buffer=5000):
    random.shuffle(file_paths)
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    ds = ds.map(load_and_preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(shuffle_buffer).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

