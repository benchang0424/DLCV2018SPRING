import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import time
import math
import numpy as np
import pandas as pd
import scipy.misc
import os


def read_image(filepath):
    print(filepath)
    images = [] 
    img_path_list = os.listdir(filepath)
    img_path_list.sort()

    for i, file in enumerate(img_path_list):
        img = scipy.misc.imread(os.path.join(filepath, file))
        images.append(img)

    images = np.array(images) / 255.0
    images = images.transpose(0, 3, 1, 2)
    return images


def read_image_gan(filepath_list,flip=True):
    images = [] 
    for filepath in filepath_list:
        print(filepath)
        img_path_list = os.listdir(filepath)
        img_path_list.sort()

        for i, file in enumerate(img_path_list):
            img = scipy.misc.imread(os.path.join(filepath, file))
            images.append(img)
            if(flip):
                images.append(np.fliplr(img))

    images = np.array(images) / 255.0
    images = images.transpose(0, 3, 1, 2)
    return images


def to_var(x):
    x = Variable(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


class CelebADataset(Dataset):
    """docstring for MyDataset"""
    def __init__(self, mode, train_filepath, test_filepath, train_csvpath, test_csvpath):
        #self.train_data = read_image(train_filepath)
        #self.test_data = read_image(test_filepath)     
        self.mode = mode

        if self.mode == 'VAE_train':
            self.images = read_image(train_filepath)
        if self.mode == 'VAE_test':
            self.images = read_image(test_filepath)
        if self.mode == 'GAN':
            self.images = read_image_gan([train_filepath, test_filepath],True)

        if self.mode == 'ACGAN':
            self.images = read_image_gan([train_filepath, test_filepath],False)
            self.train_attr = pd.read_csv(train_csvpath)['Smiling']
            self.test_attr = pd.read_csv(test_csvpath)['Smiling']
            self.attr = pd.concat((self.train_attr, self.test_attr), ignore_index=True)
            #self.attr = torch.FloatTensor(self.attr)
            self.attr = torch.FloatTensor(self.attr).view(-1,1,1,1)

        self.images = torch.FloatTensor(self.images)

    def __getitem__(self, index):
        data = self.images[index]
        if self.mode == 'ACGAN':
            label = self.attr[index]
            return data, label

        else: return data, data

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    TRAIN_DIR = "./hw4_data/train/"
    TEST_DIR = "./hw4_data/test/"
    TRAIN_CSVDIR = "./hw4_data/train.csv"
    TEST_CSVDIR = "./hw4_data/test.csv"

    all_imgs = read_image_gan([TRAIN_DIR,TEST_DIR],flip=True)
    print(all_imgs.shape)
