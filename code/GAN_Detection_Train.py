#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: GAN_Detection_Train.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Sun Sep 29 22:20:13 2019
#
#  Usage: python GAN_Detection_Train.py -h
#  Description: Train a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import division, print_function
import sys
from copy import deepcopy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import random
import cv2
import copy
import cycleGAN_dataset
import torch.nn as nn
import logging
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob


from torchvision import transforms, models
import pggan_dnet
from skimage.feature import graycomatrix
from torch.utils.data import DataLoader, random_split
import copy
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

# Training settings
parser.add_argument('--dataroot', type=str,
                    default='./datasets/',
                    help='path to dataset')
parser.add_argument('--training-set', default= 'horse',
                    help='The name of the training set. If leave_one_out flag is set, \
                    it is the leave-out set(use all other sets for training).')
parser.add_argument('--test-set', default='transposed_conv', type=str,
                    help='Choose test set from trainsposed_conv, nn, jpeg and resize')
parser.add_argument('--feature', default='image',
                    help='Feature used for training, choose from image and fft')
parser.add_argument('--mode', type=int, default=0, 
                    help='fft frequency band, 0: full, 1: low, 2: mid, 3: high')
parser.add_argument('--leave_one_out', action='store_true', default=False,
                    help='Test leave one out setting, using all other sets for training and test on a leave-out set.')
parser.add_argument('--jpg_level', type=str, default='90',
                    help='Test with different jpg compression effiecients, only effective when use jpg for test set.')
parser.add_argument('--resize_size', type=str, default='200', 
                    help='Test with different resize sizes, only effective when use resize for test set.')

parser.add_argument('--enable-logging',type=bool, default=False,
                    help='output to tensorlogger')
parser.add_argument('--log-dir', default='./log/',
                    help='folder to output log')
parser.add_argument('--model-dir', default='./model/',
                    help='folder to output model checkpoints')
parser.add_argument('--model', default='resnet',
                    help='Base classification model')
parser.add_argument('--num-workers', default= 1,
                    help='Number of workers to be created')
parser.add_argument('--pin-memory',type=bool, default= True,
                    help='')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=1, type=int, 
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--epochs', type=int, default=10, 
                    help='number of epochs to train (default: 10)')
parser.add_argument('--batch-size', type=int, default=64, 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32,
                    help='input batch size for testing (default: 32)')
parser.add_argument('--lr', type=float, default=0.01, 
                    help='learning rate (default: 0.01)')
parser.add_argument('--lr-decay', default=1e-2, type=float, 
                    help='learning rate decay ratio (default: 1e-6')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: SGD)')
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='Use data augmentation or not')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed (default: -1)')
parser.add_argument('--interval', type=int, default=5,
                    help='logging interval, epoch based. (default: 5)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.data_augment:
    suffix = suffix + '_da'
if args.leave_one_out:
    suffix = suffix + '_oo'
if args.feature != 'image':
    suffix = suffix + '_{}_{}'.format(args.feature, args.mode)

suffix = suffix + '_{}'.format(args.model)

if args.test_set == 'transposed_conv':
    #Use a small set to save the inferring time. Use the best model to test all the subsets in test phase. 
    dataset_names = ['satellite']
    #dataset_names = ['horse', 'zebra', 'summer', 'winter', 'apple', 'orange',
    #            'facades', 'cityscapes', 'satellite', 
    #            'ukiyoe', 'vangogh', 'cezanne', 'monet', 'photo', 'celeba_stargan']  

if args.test_set == 'nn':
    dataset_names = ['horse_nn',  'zebra_nn', 'summer_nn', 'winter_nn', 'apple_nn', 'orange_nn']

elif args.test_set == 'jpg':
    dataset_names = ['horse_jpg_{}'.format(args.jpg_level), 'zebra_jpg_{}'.format(args.jpg_level),
            'summer_jpg_{}'.format(args.jpg_level), 'winter_jpg_{}'.format(args.jpg_level),
            'apple_jpg_{}'.format(args.jpg_level), 'orange_jpg_{}'.format(args.jpg_level)]  

elif args.test_set == 'resize':
    dataset_names = ['horse_resize_{}'.format(args.resize_size), 'zebra_resize_{}'.format(args.resize_size),
            'summer_resize_{}'.format(args.resize_size), 'winter_resize_{}'.format(args.resize_size),
            'apple_resize_{}'.format(args.resize_size), 'orange_resize_{}'.format(args.resize_size)]  

# set the device to use by setting CUDA_VISIBLE_DEVICES env variable in
# order to prevent any memory allocation on unused GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    cudnn.benchmark = True
    # set random seeds
    if args.seed>-1:
        torch.cuda.manual_seed_all(args.seed)

# set random seeds
if args.seed>-1:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# create loggin directory
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

args.class_names = ['fake', 'real']


class GANDataset(cycleGAN_dataset.cycleGAN_dataset):
    """
    GANDataset to read images.  
    """
    def __init__(self, train=True, transform=None, batch_size = None, *arg, **kw):
        super(GANDataset, self).__init__(train=train, *arg, **kw)
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
    
    def __getitem__(self, index):
        def transform_img(img):
            if self.transform != None:
                img = self.transform(img.numpy())
            return img
        
        img = self.data[index]
        label = self.labels[index]

        if self.train:
            #data augmentation for training
            if args.data_augment:
                if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                    random_x = random.randint(0,32)
                    random_y = random.randint(0,32)
                    im = deepcopy(img.numpy()[random_y:(random_y+224),\
                            random_x:(random_x+224),:])
                elif args.model == 'pggan':
                    im = deepcopy(img.numpy())
            else:
                if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                    im = deepcopy(img.numpy()[16:240,16:240,:])
                elif args.model == 'pggan':
                    im = deepcopy(img.numpy())
        #centre crop for test
        else:
            if args.model == 'resnet' or args.model == 'densenet' or args.model == 'googlenet':
                im = deepcopy(img.numpy()[16:240,16:240,:])
            elif args.model == 'pggan':
                im = deepcopy(img.numpy())

        #use spectrum
        if args.feature == 'fft':
            im = im.astype(np.float32)
            im = im/255.0
            for i in range(3):
                img = im[:,:,i]
                fft_img = np.fft.fft2(img)
                fft_img = np.log(np.abs(fft_img)+1e-3)
                fft_min = np.percentile(fft_img,5)
                fft_max = np.percentile(fft_img,95)
                fft_img = (fft_img - fft_min)/(fft_max - fft_min)
                fft_img = (fft_img-0.5)*2
                fft_img[fft_img<-1] = -1
                fft_img[fft_img>1] = 1
                #set mid and high freq to 0
                if args.mode>0:
                    fft_img = np.fft.fftshift(fft_img)
                    if args.mode == 1:
                        fft_img[:57, :] = 0
                        fft_img[:, :57] = 0
                        fft_img[177:, :] = 0
                        fft_img[:, 177:] = 0
                    #set low and high freq to 0
                    elif args.mode == 2:
                        fft_img[:21, :] = 0
                        fft_img[:, :21] = 0
                        fft_img[203:, :] = 0
                        fft_img[:, 203:] = 0
                        fft_img[57:177, 57:177] = 0
                    #set low and mid freq to 0
                    elif args.mode == 3:
                        fft_img[21:203, 21:203] = 0
                    fft_img = np.fft.fftshift(fft_img)
                im[:,:,i] = fft_img
        else:
            im = im.astype(np.float32)
            im = (im/255 - 0.5)*2
            #img = transform_img(img)
        im = np.transpose(im, (2,0,1))
        return (im, label)

    def __len__(self):
        return self.labels.size(0)

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    # Load full training dataset
    full_train_dataset = GANDataset(
        train=True,
        batch_size=args.batch_size,
        root=args.dataroot,
        name=args.training_set,
        check_cached=args.check_cached,
        leave_one_out=args.leave_one_out,
        transform=transform
    )

    # Split into train (80%) and validation (20%)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loaders = [{'name': name,
                     'dataloader': DataLoader(
                         GANDataset(train=False,
                                    leave_one_out=False,
                                    batch_size=args.test_batch_size,
                                    root=args.dataroot,
                                    name=name,
                                    check_cached=args.check_cached,
                                    transform=transform),
                         batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, val_loader, test_loaders


def train(train_loader, val_loader, model, optimizer, criterion, epoch, logger):
    # Enable logging
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')

    # Switch to train mode
    model.train()
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for batch_idx, data in pbar:
        image_pair, label = data

        if args.cuda:
            image_pair, label = image_pair.cuda(), label.cuda()

        image_pair, label = Variable(image_pair), Variable(label)

        optimizer.zero_grad()  # Clear gradients
        out = model(image_pair)
        loss = criterion(out, label)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        train_loss += loss.item()
        logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")  # Log the loss
        pbar.set_description(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)  # Compute average training loss
    adjust_learning_rate(optimizer)  # Adjust learning rate

    # Validation phase
    model.eval()  # Set to evaluation mode
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculations
        for images, labels in val_loader:
            if args.cuda:
                images, labels = images.cuda(), labels.cuda()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)  # Compute average validation loss
    val_accuracy = correct / total * 100  # Compute validation accuracy

    # Log losses and accuracy
    logging.info(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    if args.enable_logging:
        logger.log_value('train_loss', train_loss).step()
        logger.log_value('val_loss', val_loss).step()
        logger.log_value('val_acc', val_accuracy).step()

    # Save model checkpoint every 10 epochs
    os.makedirs(f"{args.model_dir}{suffix}", exist_ok=True)
    if (epoch + 1) % 10 == 0:
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()},
                   f"{args.model_dir}{suffix}/checkpoint_{epoch+1}.pth")

def test(test_loader, model, epoch, logger, logger_test_name):
    # evaluates the model on a test dataset
    # calculates the accuracy
    # logs the results
    model.eval()

    all_preds = []
    all_labels = []

    labels, predicts = [], []
    outputs = []
    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (image_pair, label) in pbar:
        if args.cuda:
            image_pair = image_pair.cuda()
        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)
        out = model(image_pair)
        _, pred = torch.max(out,1)
        # Store predictions and true labels for classification report
        preds = torch.argmax(out, dim=1)  # Convert logits to class predictions
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        ll = label.data.cpu().numpy().reshape(-1, 1)
        pred = pred.data.cpu().numpy().reshape(-1, 1)
        out = out.data.cpu().numpy().reshape(-1, 2)
        labels.append(ll)
        predicts.append(pred)
        outputs.append(out)

    # calculation and saving performance metrics
    performance_metrics(all_labels, all_preds, epoch)
    display_random_test_samples(image_pair, all_labels, all_preds)

    num_tests = test_loader.dataset.labels.size(0)
    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)
    outputs = np.vstack(outputs).reshape(num_tests,2)

    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts)/float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))
    
    if (args.enable_logging):
        logger.log_value(logger_test_name+' Acc', acc)
    return

def performance_metrics(all_labels, all_preds, epoch):
    # Generate and log classification report
    class_report = classification_report(all_labels, all_preds, target_names=args.class_names)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    logging.info(f"Classification Report:\n{class_report}")
    print(f"Classification Report:\n{class_report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save the classification report to a file
    report_file_path = os.path.join(args.model_dir, "classification_report_test.txt")
    # Append the results to the file for each epoch
    with open(report_file_path, 'a') as f:
        f.write(f"Epoch {epoch + 1}/{args.epochs}\n")
        f.write(f"Classification Report:\n{class_report}\n")
        f.write(f"Confusion Matrix:\n{conf_matrix}\n")
        f.write("\n" + "=" * 50 + "\n")  # Add a separator between epochs for clarity

    # Visualize the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=args.class_names, yticklabels=args.class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(args.model_dir, "confusion_matrix.png"))
    plt.close()

def display_random_test_samples(images, labels, preds):
    #image, labels = read_image_file(args.data_dir, args.dataset_name, 0)
    # Select five random indices
    imggs, labs = read_test_images()
    random_indices = random.sample(range(len(images)), 5)

    # Plot the images with their ground truth and predicted labels
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(random_indices):
        img_spec = images[idx].cpu().permute(1, 2, 0).numpy()  # Convert tensor to numpy (H, W, C)
        img = cv2.cvtColor(imggs[idx], cv2.COLOR_BGR2RGB)
        true_label = args.class_names[labels[idx].item()]
        predicted_label = args.class_names[preds[idx].item()]

        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.subplot(2, 5, i + 6)
        plt.imshow(img_spec)
        plt.axis('off')
    plt.show()

def read_test_images():
    """
    Reads images from the specified directories and returns image and label arrays.
    :param data_dir: Base directory containing the dataset
    :param dataset_name: Name of the dataset
    :return: Tuple of numpy arrays (images, labels)
    """
    image_list = []
    label_list = []
    data_dir = args.dataroot
    dataset_name = 'satellite'

    # Define the search patterns for real and fake images
    search_patterns = [
        (f'{data_dir}/real/{dataset_name}/testA/*.png', 1),
        (f'{data_dir}/fake/{dataset_name}/testA/*.jpg', 0),
        (f'{data_dir}/real/{dataset_name}/testB/*.png', 1),
        (f'{data_dir}/fake/{dataset_name}/testB/*.jpg', 0)
    ]

    for search_str, label in search_patterns:
        print(f'Searching: {search_str}')
        for filename in glob.glob(search_str):
            try:
                image = cv2.imread(filename)
                if image is None:
                    print(f"Warning: Unable to read {filename}. Skipping.")
                    continue
                # Resize image to 256x256 if needed
                if image.shape[:2] != (256, 256):
                    image = cv2.resize(image, (256, 256))
                image_list.append(image)
                label_list.append(label)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Convert lists to numpy arrays
    images = np.array(image_list, dtype=np.uint8)
    labels = np.array(label_list, dtype=np.int32)

    return images, labels


def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0.
        else:
            group['step'] += 1.
        #group['lr'] = args.lr*((1-args.lr_decay)**group['step'])
        group['lr'] = args.lr
        
    return

def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd)
    else:
        raise Exception('Not supported optimizer: {0}'.format(args.optimizer))
    return optimizer


def main(train_loader, val_loader, test_loaders, model, logger):
    print('\nparsed options:\n{}\n'.format(vars(args)))

    optimizer1 = create_optimizer(model, args.lr)
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        model.cuda()
        criterion.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('=> no checkpoint found at {}'.format(args.resume))
            
    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader,val_loader, model, optimizer1, criterion, epoch, logger)
        if epoch==(end-1):
            for test_loader in test_loaders:
                test(test_loader['dataloader'], model, epoch+1, logger, test_loader['name'])
        
if __name__ == '__main__':
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix  #creating log directory
    logger, file_logger = None, None

    pretrain_flag = not args.feature=='comatrix'
    if args.model == 'resnet':
        model = models.resnet34(pretrained=True)
        num_ftrs = model.fc.in_features  #Gets the number of input features for the fully connected (fc) layer.
        model.fc = nn.Linear(num_ftrs, 2)  #Replaces the original fully connected layer with a new one that has 2 output classes. This adapts the model for binary classification.
    elif args.model == 'pggan':
        model = pggan_dnet.SimpleDiscriminator(3, label_size=1, mbstat_avg='all',
                resolution=256, fmap_max=128, fmap_base=2048, sigmoid_at_end=False)
    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    if(args.enable_logging):
        from Loggers import Logger
        logger = Logger(LOG_DIR)
    train_loader, val_loader, test_loaders = create_loaders()
    main(train_loader, val_loader, test_loaders, model, logger)
