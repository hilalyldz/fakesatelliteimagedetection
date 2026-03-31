#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: cycleGAN_dataset.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Fri Sep 27 12:20:49 2019
#
#  Usage:
#  Description: Read and Cache CycleGAN image dataset 
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

import os
import errno
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import collections
from tqdm import tqdm
import random
import glob
import cv2


#import pdb

class cycleGAN_dataset(data.Dataset):
    def __init__(self, root, name, train=True, leave_one_out = False, transform=None, check_cached=False):
        self.image_dir = r"/dss/dsshome1/09/di97zeq/Desktop/FFT_ResNet/CMF_Data/"
        self.root = r"/dss/dsshome1/09/di97zeq/Desktop/FFT_ResNet/CMF_Data/"
        self.name = name
        self.data_dir = os.path.join(self.image_dir, name)

        self.train = train
        self.leave_one_out = leave_one_out
        self.transform = transform
        if 'auto' in name:
            self.full_list = ['horse_auto', 'zebra_auto', 'apple_auto', 'orange_auto', 'winter_auto', 'summer_auto', 
                'facades_auto', 'cityscapes_auto', 'satellite_auto', 
                'fold6_auto', 'fold7_auto', 'fold8_auto', 'fold9_auto']
        else:
            self.full_list = ['satellite']

        name_list = name.split("+")
        self.data = None
        self.labels = None
        real_name_list = []
        if not self.leave_one_out:
            real_name_list = name_list
        else:
            for name in self.full_list:
                if name not in name_list:
                    real_name_list.append(name)
        for name in real_name_list:
            if train:
                data_file = os.path.join(self.root, '{}_train.pt'.format(name))
            else:
                data_file = os.path.join(self.root, '{}_test.pt'.format(name))
            self.cache_data(data_file, name, check_cached)
            data, labels = torch.load(data_file)

            if self.data is None:
                self.data = data
                self.labels = labels
            else:
                self.data = np.concatenate((self.data, data), axis=0)
                self.labels = np.concatenate((self.labels, labels), axis=0)

        self.data = torch.ByteTensor(self.data)
        self.labels = torch.LongTensor(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    def _check_datafile_exists(self,data_file):
        return os.path.exists(data_file)

    def cache_data(self, data_file, name, check_cached):
        if check_cached:
            if self._check_datafile_exists(data_file):
                print('# Found cached data {}'.format(data_file))
                #return

        # process and save as torch files
        print('# Caching data {}'.format(data_file))

        dataset = (
            read_image_file(self.image_dir, name, self.train)
        )

        with open(data_file, 'wb') as f:
            torch.save(dataset, f, pickle_protocol=5)

def read_image_file(data_dir, dataset_name, train_flag):
    """Return a Tensor containing the patches"""
    
    image_list = []
    label_list = []
    
    if train_flag:
        real_path = f"{data_dir}/real/{dataset_name}/train/"
        fake_path = f"{data_dir}/fake/{dataset_name}/train/"
    else:
        real_path = f"{data_dir}/real/{dataset_name}/test/"
        fake_path = f"{data_dir}/fake/{dataset_name}/test/"

    extensions = ["jpg", "png"]

    # REAL images (label = 1)
    for ext in extensions:
        search_str = f"{real_path}*.{ext}"
        print(f"Search string : {search_str}")
        
        for filename in glob.glob(search_str):
            image = cv2.imread(filename)
            if image.shape[0] != 256:
                image = cv2.resize(image, (256,256))
            
            image_list.append(image)
            label_list.append(1)

    # FAKE images (label = 0)
    for ext in extensions:
        search_str = f"{fake_path}*.{ext}"
        print(f"Search string : {search_str}")
        
        for filename in glob.glob(search_str):
            image = cv2.imread(filename)
            if image.shape[0] != 256:
                image = cv2.resize(image, (256,256))
            
            image_list.append(image)
            label_list.append(0)

    return np.array(image_list), np.array(label_list)





