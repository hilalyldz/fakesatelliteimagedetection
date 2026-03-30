#!/usr/bin/python
# -*- coding: utf-8 -*-
# ===========================================================
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
# ===========================================================

import os
import numpy as np
import torch
import torch.utils.data as data
import glob
import cv2


class cycleGAN_dataset(data.Dataset):
    def __init__(self, root, name, train=True, leave_one_out=False, transform=None, check_cached=False):
        self.image_dir = r"./datasets/"
        self.root = os.path.expanduser(root)
        self.name = name
        self.data_dir = os.path.join(self.image_dir, name)

        self.train = train
        self.leave_one_out = leave_one_out
        self.transform = transform

        if train:
            data_file = os.path.join(self.root, '{}_train.pt'.format(name))
        else:
            data_file = os.path.join(self.root, '{}_test.pt'.format(name))

        dataset = (
            read_image_file(self.image_dir, name, self.train)
        )

        with open(data_file, 'wb') as f:
            torch.save(dataset, f)

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


def read_image_file(data_dir, dataset_name, train_flag):
    """Return a Tensor containing the patches
    """
    image_list = []
    label_list = []
    # load all possible jpg or png images
    if train_flag:
        search_str = '{}/real/train/*.png'.format(data_dir, dataset_name)
    else:
        search_str = '{}/real/test/*.png'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        if image.shape[0] != 256:
            image = cv2.resize(image, (256, 256))
        image_list.append(image)
        label_list.append(1)

    if train_flag:
        search_str = '{}fake/train/*.jpg'.format(data_dir, dataset_name)
    else:
        search_str = '{}/fake/test/*.jpg'.format(data_dir, dataset_name)

    for filename in glob.glob(search_str):
        image = cv2.imread(filename)
        image_list.append(image)
        label_list.append(0)

    return np.array(image_list), np.array(label_list)





