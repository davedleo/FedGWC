import os
import pickle
import imageio
import numpy as np
import torchvision.datasets as datasets





def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
    return dict





print('#### Setting up CIFAR100 ####')
datasets.CIFAR100('./datasets/CIFAR100/data/raw/images', download = True)
os.makedirs('./datasets/CIFAR100/data/raw/train', exist_ok = True)
os.makedirs('./datasets/CIFAR100/data/raw/test', exist_ok = True)

train = unpickle('./datasets/CIFAR100/data/raw/images/cifar-100-python/train')
filenames = [t.decode('utf8') for t in train[b'filenames']]
data = train[b'data']

print("Saving train images...")
for d, filename in zip(data, filenames):
    image = np.zeros((32, 32, 3), dtype = np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) 
    image[...,1] = np.reshape(d[1024:2048], (32,32)) 
    image[...,2] = np.reshape(d[2048:], (32,32)) 
    imageio.imwrite('./datasets/CIFAR100/data/raw/train/' + filename, image)

test = unpickle('./datasets/CIFAR100/data/raw/images/cifar-100-python/test')
filenames = [t.decode('utf8') for t in test[b'filenames']]
data = test[b'data']

print("Saving test images...")
for d, filename in zip(data, filenames):
    image = np.zeros((32, 32, 3), dtype = np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) 
    image[...,1] = np.reshape(d[1024:2048], (32,32)) 
    image[...,2] = np.reshape(d[2048:], (32,32)) 
    imageio.imwrite('./datasets/CIFAR100/data/raw/test/' + filename, image)

print('#### Finish ####\n')