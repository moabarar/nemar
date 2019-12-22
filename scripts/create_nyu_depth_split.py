from sklearn.model_selection import train_test_split
import h5py
import numpy as np
import joblib
from PIL import Image
import os
from tqdm import tqdm


dataset_path = '../datasets/nyu_depth'

print('Loading mat file')
file_path = '../datasets/nyu_depth/nyu_depth_v2_labeled.mat'
with h5py.File(file_path, 'r') as f:
    images = np.array(f['images'])
    depths = np.array(f['depths'])

num_imgs = images.shape[0]

print('Splitting indices')
train_inds, test_inds = train_test_split(list(range(num_imgs)))
val_inds, test_inds = train_test_split(test_inds, test_size=0.5)


train_path = os.path.join(dataset_path, 'train')
for train_ind in tqdm(train_inds, desc='Creating train files'):

    img = Image.fromarray(np.moveaxis(images[train_ind].astype(np.uint8), [0, 1, 2], [2, 1, 0]))
    depth = Image.fromarray(np.moveaxis(depths[train_ind].astype(np.uint8), [0, 1], [1, 0]))
    img.save(os.path.join(train_path, 'trainA/{}.jpg'.format(train_ind)))
    depth.save(os.path.join(train_path, 'trainB/{}.jpg'.format(train_ind)))
    a = 5


test_path = os.path.join(dataset_path, 'test')
for test_ind in tqdm(test_inds, desc='Creating test files'):

    img = Image.fromarray(np.moveaxis(images[test_ind].astype(np.uint8), [0, 1, 2], [2, 1, 0]))
    depth = Image.fromarray(np.moveaxis(depths[test_ind].astype(np.uint8), [0, 1], [1, 0]))
    img.save(os.path.join(test_path, 'testA/{}.jpg'.format(test_ind)))
    depth.save(os.path.join(test_path, 'testB/{}.jpg'.format(test_ind)))
    a = 5

val_path = os.path.join(dataset_path, 'val')
for val_ind in tqdm(val_inds, desc='Creating val files'):

    img = Image.fromarray(np.moveaxis(images[val_ind].astype(np.uint8), [0, 1, 2], [2, 1, 0]))
    depth = Image.fromarray(np.moveaxis(depths[val_ind].astype(np.uint8), [0, 1], [1, 0]))
    img.save(os.path.join(val_path, 'valA/{}.jpg'.format(val_ind)))
    depth.save(os.path.join(val_path, 'valB/{}.jpg'.format(val_ind)))
    a = 5


joblib.dump((train_inds, test_inds, val_inds), '../datasets/nyu_depth/training_split_inds.joblib')
