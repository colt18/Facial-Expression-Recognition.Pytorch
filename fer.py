''' Fer2013 Dataset class'''

from __future__ import print_function
from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data

class FER2013(data.Dataset):
 
    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        # self.data = h5py.File('./data/data.h5', 'r', driver='core')

        
        # now load the picked numpy arrays
        if self.split == 'Training':
            self.train_data = np.load('data/Training_x.npy')
            self.train_labels = np.load('data/Training_y.npy')
            # self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))
        elif self.split == 'PublicTest':
            self.PublicTest_data = np.load('data/PublicTest_x.npy')
            self.PublicTest_labels = np.load('data/PublicTest_y.npy')
            # self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.PublicTest_data = self.PublicTest_data.reshape((3589, 48, 48))

        else:
            self.PrivateTest_data = np.load('data/PrivateTest_x.npy')
            self.PrivateTest_labels = np.load('data/PrivateTest_y.npy')
            # self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.PrivateTest_data = self.PrivateTest_data.reshape((3589, 48, 48))

    def __getitem__(self, index):
      
        if self.split == 'Training':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.split == 'PublicTest':
            img, target = self.PublicTest_data[index], self.PublicTest_labels[index]
        else:
            img, target = self.PrivateTest_data[index], self.PrivateTest_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        # img = img[:, :, np.newaxis]
        # img = np.concatenate((img, img, img), axis=2)
        # img = Image.fromarray(img)
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.train_data)
        elif self.split == 'PublicTest':
            return len(self.PublicTest_data)
        else:
            return len(self.PrivateTest_data)
