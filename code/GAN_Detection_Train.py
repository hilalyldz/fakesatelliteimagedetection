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
import pywt


from torchvision import transforms, models
import pggan_dnet
from skimage.feature import graycomatrix
from torch.utils.data import DataLoader, random_split
import copy
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

train_losses = []
val_losses = []

def get_settings():
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

    if args.test_set == 'nn':
        dataset_names = ['satellite']
    elif args.test_set == 'jpg':
        dataset_names = ['satellite']
    elif args.test_set == 'resize':
        dataset_names = ['satellite']

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
    return args, suffix, dataset_names


class GANDataset(cycleGAN_dataset.cycleGAN_dataset):
    """
    GANDataset to read images.  
    """
    def __init__(self, train=True, transform=None, batch_size = None, args=None, *arg, **kw):
        super(GANDataset, self).__init__(train=train, *arg, **kw)
        self.transform = transform
        self.train = train
        self.batch_size = batch_size
        self.args = args
    
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
            if self.args.model == 'resnet' or self.args.model == 'densenet' or self.args.model == 'googlenet':
                im = deepcopy(img.numpy()[16:240,16:240,:])
            elif args.model == 'pggan':
                im = deepcopy(img.numpy())

        #use spectrum
        if self.args.feature == 'fft':
            im = self.fast_fourier_transformation(im)
        elif self.args.feature == 'wavelet':
            im = self.wavelet_transformation(im)
        else:
            im = im.astype(np.float32)
            im = (im/255 - 0.5)*2
            #img = transform_img(img)
        fft_images = im
        if self.args.feature != 'wavelet':
            im = np.transpose(im, (2,0,1))
        #self.visualize_and_save(index, fft_images)
        return (im, label)

    def __len__(self):
        return self.labels.size(0)

    def visualize_and_save(self, index, fft_images, output_path="C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Dataset_Visualization/"):
        im = self.data[index]
        label = self.labels[index]
        path = os.path.join(output_path, f"image_{index}.png")

        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].imshow(im)
        axs[0].set_title(f"Original Image(Label: {label})")
        axs[0].axis('off')

        axs[1].imshow(np.log1p(np.abs(fft_images)), cmap="gray")
        axs[1].set_title("Frequency Spectrum")
        axs[1].axis('off')

        plt.savefig(path)
        plt.close(fig)

    def fast_fourier_transformation(self, im):
        im = im.astype(np.float32)
        im = im / 255.0
        for i in range(3):
            img = im[:, :, i]
            # === FFT Part ===
            fft_img = np.fft.fft2(img)
            fft_shifted = np.fft.fftshift(fft_img)
            # === HIGH-PASS FILTER ===
            fft_filtered = self.high_pass_filter(fft_shifted)
            # === LOG MAGNITUDE ===
            fft_filtered = np.fft.ifftshift(fft_filtered)  # shifting back
            fft_img = np.log(np.abs(fft_filtered) + 1e-3)

            fft_min = np.percentile(fft_img, 5)
            fft_max = np.percentile(fft_img, 95)
            fft_img = (fft_img - fft_min) / (fft_max - fft_min)
            fft_img = (fft_img - 0.5) * 2
            fft_img[fft_img < -1] = -1
            fft_img[fft_img > 1] = 1
            # set mid and high freq to 0
            if self.args.mode > 0:
                fft_img = np.fft.fftshift(fft_img)
                if self.args.mode == 1:
                    fft_img[:57, :] = 0
                    fft_img[:, :57] = 0
                    fft_img[177:, :] = 0
                    fft_img[:, 177:] = 0
                # set low and high freq to 0
                elif self.args.mode == 2:
                    fft_img[:21, :] = 0
                    fft_img[:, :21] = 0
                    fft_img[203:, :] = 0
                    fft_img[:, 203:] = 0
                    fft_img[57:177, 57:177] = 0
                # set low and mid freq to 0
                elif self.args.mode == 3:
                    fft_img[21:203, 21:203] = 0
                fft_img = np.fft.fftshift(fft_img)
            im[:, :, i] = fft_img
        return im

    def wavelet_transformation(self, im):
        im = im.astype(np.float32)
        im = im / 255.0  # Normalize to [0, 1]
        wavelet_channels = []
        for i in range(3):  # R, G, B
            coeffs2 = pywt.dwt2(im[:, :, i], 'haar')
            LL, (LH, HL, HH) = coeffs2
            wavelet_channels.extend([LL, LH, HL, HH])

        resized_channels = [cv2.resize(c, (224, 224), interpolation=cv2.INTER_LINEAR) for c in wavelet_channels]
        im = np.stack(resized_channels, axis=0).astype(np.float32)
        # Normalize to [-1,1]
        im_min = im.min()
        im_max = im.max()
        im = (im - im_min) / (im_max - im_min + 1e-8)
        im = (im - 0.5) * 2
        return im

    def high_pass_filter(self, fft_shifted):
        rows, cols = fft_shifted.shape
        crow, ccol = rows // 2, cols // 2
        radius = 30  # cutoff radius (tuning is necessary)
        mask = np.ones((rows, cols), dtype=np.uint8)
        cv2.circle(mask, (crow, ccol), radius, 0, -1)  # removing low frequency
        fft_filtered = fft_shifted * mask
        return fft_filtered

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    if args.feature == 'wavelet':
        transform = None
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Load full training dataset
    full_train_dataset = GANDataset(
        train=True,
        batch_size=args.batch_size,
        root=args.dataroot,
        name=args.training_set,
        check_cached=args.check_cached,
        leave_one_out=args.leave_one_out,
        transform=transform,
        args=args
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
                                    transform=transform,
                                    args=args),
                         batch_size=args.test_batch_size,
                         shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return train_loader, val_loader, test_loaders

def fft_band_masks(h, w):
    crow, ccol = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)

    low = dist <= 30
    mid = (dist > 30) & (dist <= 80)
    high = dist > 80

    return low, mid, high

def band_contribution(cam):
    h, w = cam.shape
    low, mid, high = fft_band_masks(h, w)

    scores = {
        "LOW": cam[low].mean(),
        "MID": cam[mid].mean(),
        "HIGH": cam[high].mean()
    }

    dominant_band = max(scores, key=scores.get)
    return scores, dominant_band

def fft_complex_rgb(im):
    """
    im: spatial RGB image in [0,1], shape (H,W,3)
    returns: list of complex FFTs (one per channel)
    """
    fft_channels = []
    for c in range(3):
        fft = np.fft.fftshift(np.fft.fft2(im[:, :, c]))
        fft_channels.append(fft)
    return fft_channels

def spatial_backprojection(fft_channels, cam):
    """
    fft_channels: list of complex FFTs
    cam: Grad-CAM map in frequency domain (H,W), normalized [0,1]
    returns: spatial artifact map (H,W) in [0,1]
    """
    spatial_maps = []

    for fft in fft_channels:
        weighted_fft = fft * cam
        img_back = np.fft.ifft2(np.fft.ifftshift(weighted_fft))
        spatial_maps.append(np.abs(img_back))

    spatial_map = np.mean(spatial_maps, axis=0)
    spatial_map = (spatial_map - spatial_map.min()) / \
                  (spatial_map.max() - spatial_map.min() + 1e-8)
    return spatial_map

def train(train_loader, val_loader, model, optimizer, criterion, epoch, logger):
    global train_losses, val_losses
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

        image_pair, label = Variable(image_pair), Variable(label) # It can be removed

        optimizer.zero_grad()  # Clear gradients
        out = model(image_pair)
        loss = criterion(out, label)
        loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        train_loss += loss.item()
        logging.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")  # Log the loss
        pbar.set_description(f"Batch {batch_idx} Loss: {loss.item():.4f}")

    train_loss /= len(train_loader)  # Compute average training loss
    train_losses.append(train_loss)  # All train losses
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
            images, labels = Variable(images), Variable(labels) # It can be removed

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    val_loss /= len(val_loader)  # Compute average validation loss
    val_losses.append(val_loss)  # All validation losses
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

    cam_cache = []
    MAX_CAM_SAMPLES = 6
    global_idx = 0

    # Create CAM directory per epoch
    if not os.path.exists(
            f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Spectral_Explainability/cam_epoch_{epoch}"):
        os.makedirs(f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Spectral_Explainability/cam_epoch_{epoch}")
    # Set up Grad Cam for ResNet model
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    cam = None
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

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

        # Cached data for Grad-CAM
        for i in range(image_pair.size(0)):
            if len(cam_cache) < MAX_CAM_SAMPLES:
                cam_cache.append({
                    "tensor": image_pair[i].detach().cpu(),
                    "gt": label[i].item(),  # ground truth
                    "pred": pred[i].item(),  # model prediction
                    "index": global_idx
                })
            else:
                j =random.randint(0, global_idx)
                if j < MAX_CAM_SAMPLES:
                    cam_cache = {
                    "tensor": image_pair[i].detach().cpu(),
                    "gt": label[i].item(),  # ground truth
                    "pred": pred[i].item(),  # model prediction
                    "index": global_idx
                }
            global_idx += 1

    band_stats = {
        "real": {"LOW": [], "MID": [], "HIGH": []},
        "fake": {"LOW": [], "MID": [], "HIGH": []}
    }
    spatial_images, _ = read_test_images()
    # Grad-CAM visualization — only for a few samples
    # === FFT Grad-CAM (FAKE class) ===
    for k, sample in enumerate(cam_cache):
        gt_label = args.class_names[sample["gt"]]  # REAL / FAKE
        pred_label = args.class_names[sample["pred"]]  # REAL / FAKE
        input_tensor = sample["tensor"].unsqueeze(0).to(device)

        # 0 = FAKE, 1 = REAL (according to your class order)
        target = [ClassifierOutputTarget(0)]
        cam_class = "FAKE_CAM"

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=target
        )[0]

        # Normalize CAM
        cam_norm = (grayscale_cam - grayscale_cam.min()) / \
                   (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

        # Band analysis
        scores, dominant_band = band_contribution(cam_norm)
        for band, score in scores.items():
            band_stats[gt_label][band].append(score)

        print(f"[GradCAM] Sample-{k}")
        print(f"  GT   : {gt_label}")
        print(f"  Pred : {pred_label}")
        print(f"  Band scores: {scores}")
        print(f"  FAKE decision dominated by: {dominant_band}")
        print("\n=== Average FAKE Grad-CAM Band Contribution ===")
        for class_name in band_stats:  # 'real' / 'fake'
            print(f"\nClass: {class_name.upper()}")
            for band in band_stats[class_name]:  # 'LOW', 'MID', 'HIGH'
                values = band_stats[class_name][band]
                avg = np.mean(values) if values else 0.0
                print(f"  {band}: {avg:.4f}")

        # ----- Load corresponding spatial image -----
        spatial_img = spatial_images[sample["index"]]
        spatial_img = cv2.cvtColor(spatial_img, cv2.COLOR_BGR2RGB)
        spatial_img = cv2.resize(spatial_img, (224, 224))
        spatial_img = spatial_img.astype(np.float32) / 255.0

        # ----- FFT → spatial backprojection -----
        fft_channels = fft_complex_rgb(spatial_img)
        spatial_map = spatial_backprojection(fft_channels, cam_norm)

        # ----- Overlay spatial artifact map -----
        overlay = show_cam_on_image(
            spatial_img,
            spatial_map,
            use_rgb=True
        )

        # ----- Save results -----
        base_path = (
            f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/"
            f"Spectral_Explainability/cam_epoch_{epoch}/"
            f"GT-{gt_label}_PRED-{pred_label}_IDX-{sample['index']}"
        )

        cv2.imwrite(f"{base_path}_fft_cam.png", (cam_norm * 255).astype(np.uint8))
        cv2.imwrite(f"{base_path}_spatial_projection.png", (overlay * 255).astype(np.uint8))

        #First version of gradcam
       # # Save frequency CAM
       # cam_uint8 = (cam_norm * 255).astype(np.uint8)
       # save_path = (
       #     f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/"
       #     f"Cam_Results/cam_epoch_{epoch}/"
       #     f"fft_cam_{batch_idx}_{i}_{dominant_band}.png"
       # )
       # cv2.imwrite(save_path, cam_uint8)

    '''
    if cam is not None: #and batch_idx < 5:  # limit to first 5 batches
        if args.feature == 'wavelet':
            for i in range(min(image_pair.size(0), 2)):  # visualize max 2 images per batch
                input_tensor = image_pair[i].unsqueeze(0)
                target = [ClassifierOutputTarget(label[i].item())]

                grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=target)
                grayscale_cam = grayscale_cam[0]

                # Convert image to visualizable format
                # Assuming you want to show only LL bands (channels 0–2)
                img_np = image_pair[i, :3, :, :].detach().cpu().numpy()
                img_np = img_np.transpose(1, 2, 0)
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

                cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

                # Save the CAM image
                cam_path = f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Cam_Results/sample_{batch_idx}_{i}.png"
                cv2.imwrite(cam_path, cam_image)
        elif 1:#args.feature == 'fft':
            for i in range(min(image_pair.size(0), 2)):
                input_tensor = image_pair[i].unsqueeze(0)

                # Target class (ground truth)
                target = [ClassifierOutputTarget(label[i].item())]

                # Generate Grad-CAM (still on FFT input if model was trained on FFT)
                grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=target)[0]

                img_spatial = image_pair[i].detach().cpu().numpy()  # (channels,H,W)
                img_spatial = img_spatial.transpose(1, 2, 0)  # (H,W,channels)
                img_spatial = (img_spatial - img_spatial.min()) / (img_spatial.max() - img_spatial.min() + 1e-8)

                # Overlay CAM on spatial image
                cam_image = show_cam_on_image(img_spatial, grayscale_cam, use_rgb=True)
                cam_image_uint8 = (cam_image * 255).astype(np.uint8)

                # Save overlay
                cam_path = f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Cam_Results/spatial_overlay_{batch_idx}_{i}.png"
                cv2.imwrite(cam_path, cam_image_uint8)

                # Optional: save raw CAM heatmap
                heatmap_path = f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Cam_Results/spatial_heatmap_{batch_idx}_{i}.png"
                heatmap_uint8 = (grayscale_cam * 255).astype(np.uint8)
                cv2.imwrite(heatmap_path, heatmap_uint8)

                # Original image save
                original_path = f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Cam_Results/original_image_{batch_idx}_{i}.png"
                original_uint8 = (img_spatial * 255).astype(np.uint8)
                cv2.imwrite(original_path, original_uint8)
    '''

    # calculation and saving performance metrics
    performance_metrics(all_labels, all_preds, epoch)
    #display_random_test_samples(image_pair, all_labels, all_preds)

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
        (f'{data_dir}/real/{dataset_name}/test/*.jpg', 1),
        (f'{data_dir}/fake/{dataset_name}/test/*.jpg', 0)
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

def plot_losses(save_path="loss_curve.png"):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses) + 1), val_losses,
             label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()   # <-- important: prevents showing / memory leak


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

    # Set device
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)

    optimizer1 = create_optimizer(model, args.lr)
    criterion = nn.CrossEntropyLoss()
    #if args.cuda:
    #    model.cuda()
    #    criterion.cuda()

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
    #for test_loader in test_loaders:
    #    test(test_loader['dataloader'], model, 0, logger, test_loader['name'])
    for epoch in range(start, end):
        # iterate over test loaders and test results
        train(train_loader,val_loader, model, optimizer1, criterion, epoch, logger)
        #if epoch==(end-1):
        #    for test_loader in test_loaders:
        #        test(test_loader['dataloader'], model, epoch+1, logger, test_loader['name'])
    for test_loader in test_loaders:
        test(test_loader['dataloader'], model, end-1, logger, test_loader['name'])
    plot_losses()
        
if __name__ == '__main__':
    args, suffix, dataset_names = get_settings()
    LOG_DIR = args.log_dir
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    LOG_DIR = args.log_dir + suffix  #creating log directory
    logger, file_logger = None, None

    pretrain_flag = not args.feature=='comatrix'
    if args.model == 'resnet':
        if args.feature == 'wavelet':
            model = models.resnet34(pretrained=False)
            new_input_channels = 12  # because you use LL, LH, HL, HH for R, G, B
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels=new_input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            # Optionally copy weights from 3-channel model
            with torch.no_grad():
                model.conv1.weight[:, :3] = original_conv.weight
                for i in range(3, new_input_channels):
                    model.conv1.weight[:, i] = original_conv.weight[:, i % 3]

            model.fc = nn.Linear(model.fc.in_features, 2)
        else:
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
