import setup
import keras
from keras import layers
import tensorflow as tf
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

def load_data():
    url = "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
    path = keras.utils.get_file(origin=url)
    data = np.load(path)
    
    images = data["images"]
    poses = data["poses"]
    focal = data["focal"]

    return images, poses, focal

def build_datasets(images, poses, focal):
    # your existing dataset code verbatim
    return train_ds, val_ds

def train_nerf(train_ds, val_ds):
    # model creation + compile + fit
    return model

def render_results(model, ...):
    # gif + video code
    return