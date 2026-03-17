#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: GAN_Detection_Train.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct 15 16:41:30 2019
#
#  Usage: python GAN_Detection_Test.py -h
#  Description: Evaluate a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================

from __future__ import division, print_function
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import os
from tqdm import tqdm
import numpy as np
import cv2
import copy
from GAN_Detection_Train import GANDataset
import torch.nn as nn
from collections import OrderedDict
import csv
from sklearn.metrics import classification_report, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pywt

from torchvision import transforms, models

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

# Training settings
parser.add_argument('--dataroot', type=str,
                    default=r'C:\Users\yild_hi\PycharmProjects\fakesatelliteimagedetection1\CMF_Data_original',
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

parser.add_argument('--result-dir', default='./final_output/',
                    help='folder to output result in csv')
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
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--data_augment', action='store_true', default=False,
                    help='Use data augmentation or not')
parser.add_argument('--check_cached', action='store_true', default=True,
                    help='Use cached dataset or not')
parser.add_argument('--seed', type=int, default=-1, 
                    help='random seed (default: -1)')

# Device options
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

suffix = '{}'.format(args.training_set)

if args.test_set == 'transposed_conv':
    dataset_names = ['satellite']
    #dataset_names = ['horse+zebra', 'apple+orange', 'summer+winter', 'facades', 'cityscapes', 'satellite', 'fold6', 'fold7', 'fold8', 'fold9']
elif args.test_set == 'nn':
    dataset_names = ['horse_nn',  'zebra_nn', 'summer_nn', 'winter_nn', 'apple_nn', 'orange_nn', 'horse', 'zebra', 'summer', 'winter', 'apple', 'orange']
elif args.test_set == 'jpg':
    dataset_names = ['horse_jpg_{}'.format(args.jpg_level), 'zebra_jpg_{}'.format(args.jpg_level),
            'summer_jpg_{}'.format(args.jpg_level), 'winter_jpg_{}'.format(args.jpg_level),
            'apple_jpg_{}'.format(args.jpg_level), 'orange_jpg_{}'.format(args.jpg_level),
            'facades_jpg_{}'.format(args.jpg_level), 'cityscapes_jpg_{}'.format(args.jpg_level),
            'satellite_jpg_{}'.format(args.jpg_level), 'ukiyoe_jpg_{}'.format(args.jpg_level),
            'vangogh_jpg_{}'.format(args.jpg_level), 'cezanne_jpg_{}'.format(args.jpg_level),
            'monet_jpg_{}'.format(args.jpg_level), 'photo_jpg_{}'.format(args.jpg_level)]  
elif args.test_set == 'resize':
    dataset_names = ['horse_resize_{}'.format(args.resize_size), 'zebra_resize_{}'.format(args.resize_size),
            'summer_resize_{}'.format(args.resize_size), 'winter_resize_{}'.format(args.resize_size),
            'apple_resize_{}'.format(args.resize_size), 'orange_resize_{}'.format(args.resize_size),
            'facades_resize_{}'.format(args.resize_size), 'cityscapes_resize_{}'.format(args.resize_size),
            'satellite_resize_{}'.format(args.resize_size), 'ukiyoe_resize_{}'.format(args.resize_size),
            'vangogh_resize_{}'.format(args.resize_size), 'cezanne_resize_{}'.format(args.resize_size),
            'monet_resize_{}'.format(args.resize_size), 'photo_resize_{}'.format(args.resize_size)]  
else:
    print('Test set does not support!')
    exit(-1)
    

if args.leave_one_out:
    args.training_set = args.training_set.replace('_auto','')
    dataset_names = [args.training_set]

if args.data_augment:
    suffix = suffix + '_da'
if args.leave_one_out:
    suffix = suffix + '_oo'
if args.feature != 'image':
    suffix = suffix + '_{}_{}'.format(args.feature, args.mode)

suffix = suffix + '_{}'.format(args.model)

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

try:
    os.stat('{}/'.format(args.result_dir))
except:
    os.makedirs('{}/'.format(args.result_dir))

args.class_names = ['fake', 'real']

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

def wavelet_backprojection(wavelet_tensor, cam):
    """
    wavelet_tensor: (12,H,W) numpy array
    cam: GradCAM map (H,W) normalized [0,1]

    returns: spatial artifact map (H,W)
    """

    spatial_maps = []

    for c in range(3):  # R, G, B
        LL = wavelet_tensor[c]
        LH = wavelet_tensor[c+3]
        HL = wavelet_tensor[c+6]
        HH = wavelet_tensor[c+9]

        # Weight coefficients using CAM
        LL_w = LL * cam
        LH_w = LH * cam
        HL_w = HL * cam
        HH_w = HH * cam

        # Inverse wavelet
        coeffs = (LL_w, (LH_w, HL_w, HH_w))
        reconstructed = pywt.idwt2(coeffs, 'haar')

        spatial_maps.append(np.abs(reconstructed))

    spatial_map = np.mean(spatial_maps, axis=0)

    spatial_map = (spatial_map - spatial_map.min()) / \
                  (spatial_map.max() - spatial_map.min() + 1e-8)

    return spatial_map

def wavelet_band_contribution(wavelet_tensor, cam):
    """
    wavelet_tensor: (12, H, W)
    cam: (H, W)

    returns:
        band_scores: dict
        dominant_band: str
    """

    band_scores = {
        "LL": 0.0,
        "LH": 0.0,
        "HL": 0.0,
        "HH": 0.0
    }

    for c in range(3):  # RGB

        LL = wavelet_tensor[c]
        LH = wavelet_tensor[c + 3]
        HL = wavelet_tensor[c + 6]
        HH = wavelet_tensor[c + 9]

        band_scores["LL"] += np.mean(np.abs(LL) * cam)
        band_scores["LH"] += np.mean(np.abs(LH) * cam)
        band_scores["HL"] += np.mean(np.abs(HL) * cam)
        band_scores["HH"] += np.mean(np.abs(HH) * cam)

    # average over RGB
    for key in band_scores:
        band_scores[key] /= 3.0

    dominant_band = max(band_scores, key=band_scores.get)

    return band_scores, dominant_band

def compute_mean_scores(score_list):
    mean_scores = {
        "LL": 0.0,
        "LH": 0.0,
        "HL": 0.0,
        "HH": 0.0
    }

    if len(score_list) == 0:
        return mean_scores

    for s in score_list:
        for key in mean_scores:
            mean_scores[key] += s[key]

    for key in mean_scores:
        mean_scores[key] /= len(score_list)

    return mean_scores

def create_thesis_figure(original, gradcam, spatial, overlay, save_path):

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(original)
    axs[0].set_title("(a) Original")
    axs[0].axis('off')

    axs[1].imshow(gradcam)
    axs[1].set_title("(b) Grad-CAM")
    axs[1].axis('off')

    axs[2].imshow(spatial)
    axs[2].set_title("(c) Spatial Projection")
    axs[2].axis('off')

    axs[3].imshow(overlay)
    axs[3].set_title("(d) Overlay")
    axs[3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

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

def create_loaders():

    test_dataset_names = copy.copy(dataset_names)

    kwargs = {'num_workers': args.num_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    print(test_dataset_names) 
    test_loaders = [{'name': name,
                     'dataloader': torch.utils.data.DataLoader(
             GANDataset(train=args.leave_one_out,
                     batch_size=args.test_batch_size,
                     root=args.dataroot,
                     name=name,
                     check_cached=args.check_cached,
                     transform=transform, args=args),
                        batch_size=args.test_batch_size,
                        shuffle=False, **kwargs)}
                    for name in test_dataset_names]

    return test_loaders

def test(test_loader, model, epoch, logger_test_name):

    model.eval()

    all_preds = []
    all_labels = []
    labels, predicts, outputs = [], [], []
    all_wavelet_scores = []
    real_wavelet_scores = []
    fake_wavelet_scores = []

    save_dir = f"C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/Spectral_Explainability_ALL/"
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)

    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])

    pbar = tqdm(enumerate(test_loader))

    spatial_images, _ = read_test_images()

    global_idx = 0

    for batch_idx, (image_pair, label) in pbar:

        if args.cuda:
            image_pair = image_pair.cuda()

        with torch.no_grad():
            image_pair, label = Variable(image_pair), Variable(label)

        out = model(image_pair)
        _, pred = torch.max(out, 1)

        preds = torch.argmax(out, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

        ll = label.data.cpu().numpy().reshape(-1, 1)
        pr = pred.data.cpu().numpy().reshape(-1, 1)
        out_np = out.data.cpu().numpy().reshape(-1, 2)

        labels.append(ll)
        predicts.append(pr)
        outputs.append(out_np)

        # ===============================
        # 🔥 Grad-CAM for ALL IMAGES
        # ===============================
        for i in range(image_pair.size(0)):

            input_tensor = image_pair[i].unsqueeze(0).to(device)

            gt_label = args.class_names[label[i].item()]
            pred_label = args.class_names[pred[i].item()]

            # ---- Grad-CAM ----
            grayscale_cam = cam(
                input_tensor=input_tensor,
                targets=[ClassifierOutputTarget(0)]
            )[0]

            cam_norm = (grayscale_cam - grayscale_cam.min()) / \
                       (grayscale_cam.max() - grayscale_cam.min() + 1e-8)

            # ===============================
            # 🔥 WAVELET BACKPROJECTION
            # ===============================
            wavelet_tensor = image_pair[i].cpu().numpy()

            spatial_maps = []

            for c in range(3):
                LL = wavelet_tensor[c]
                LH = wavelet_tensor[c + 3]
                HL = wavelet_tensor[c + 6]
                HH = wavelet_tensor[c + 9]

                LL_w = LL * cam_norm
                LH_w = LH * cam_norm
                HL_w = HL * cam_norm
                HH_w = HH * cam_norm
                coeffs = (LL_w, (LH_w, HL_w, HH_w))

                reconstructed = pywt.idwt2(coeffs, 'haar')
                spatial_maps.append(np.abs(reconstructed))

            spatial_map = np.mean(spatial_maps, axis=0)
            spatial_map = (spatial_map - spatial_map.min()) / \
                          (spatial_map.max() - spatial_map.min() + 1e-8)

            # ===============================
            # ORIGINAL IMAGE
            # ===============================
            spatial_img = spatial_images[global_idx]
            spatial_img = cv2.cvtColor(spatial_img, cv2.COLOR_BGR2RGB)
            spatial_img = cv2.resize(spatial_img,
                                     (spatial_map.shape[1], spatial_map.shape[0]))
            spatial_img = spatial_img.astype(np.float32) / 255.0

            overlay = show_cam_on_image(spatial_img, spatial_map, use_rgb=True)

            # ===============================
            # SAVE PATH
            # ===============================
            base_path = os.path.join(
                save_dir,
                f"IDX-{global_idx}_GT-{gt_label}_PRED-{pred_label}"
            )

            # ---- BASIC SAVES ----
            cv2.imwrite(f"{base_path}_gradcam.png",
                        (cam_norm * 255).astype(np.uint8))

            cv2.imwrite(f"{base_path}_spatial.png",
                        (spatial_map * 255).astype(np.uint8))

            cv2.imwrite(f"{base_path}_overlay.png",
                        (overlay * 255).astype(np.uint8))

            # ===============================
            # 🎨 COLORBAR FIGURES (TEZ İÇİN)
            # ===============================
            plt.figure(figsize=(4, 4))
            plt.imshow(cam_norm, cmap='jet')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{base_path}_gradcam_colorbar.png", dpi=300)
            plt.close()

            plt.figure(figsize=(4, 4))
            plt.imshow(spatial_map, cmap='jet')
            plt.colorbar()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{base_path}_spatial_colorbar.png", dpi=300)
            plt.close()

            # ===============================
            # BAND ANALYSIS
            # ===============================
            if args.feature == 'fft':
                scores, dominant_band = band_contribution(cam_norm)

            elif args.feature == 'wavelet':
                scores, dominant_band = wavelet_band_contribution(
                    wavelet_tensor,
                    cam_norm
                )
            if args.feature == 'wavelet':
                all_wavelet_scores.append(scores)
                if gt_label == 'real':
                    real_wavelet_scores.append(scores)
                elif gt_label == 'fake':
                    fake_wavelet_scores.append(scores)

            with open(f"{base_path}_bands.txt", "w") as f:
                f.write(f"GT: {gt_label}\n")
                f.write(f"Prediction: {pred_label}\n\n")
                for band in scores:
                    f.write(f"{band}: {scores[band]:.6f}\n")
                f.write(f"\nDominant: {dominant_band}\n")

            global_idx += 1

    performance_metrics(all_labels, all_preds, epoch)
    # Wavelet Average Band Contribution
    if args.feature == 'wavelet' and len(all_wavelet_scores) > 0:

        mean_scores = {
            "LL": 0.0,
            "LH": 0.0,
            "HL": 0.0,
            "HH": 0.0
        }

        for s in all_wavelet_scores:
            for key in mean_scores:
                mean_scores[key] += s[key]

        for key in mean_scores:
            mean_scores[key] /= len(all_wavelet_scores)

        print("\n=== Average Wavelet Band Contribution ===")
        for k, v in mean_scores.items():
            print(f"{k}: {v:.6f}")

        bands = list(mean_scores.keys())
        values = list(mean_scores.values())

        plt.figure(figsize=(6, 4))
        plt.bar(bands, values)
        plt.xlabel("Wavelet Bands")
        plt.ylabel("Contribution")
        plt.title("Average Wavelet Band Contribution")

        save_path = os.path.join(save_dir, "wavelet_band_distribution.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    if args.feature == 'wavelet':

        real_mean = compute_mean_scores(real_wavelet_scores)
        fake_mean = compute_mean_scores(fake_wavelet_scores)

        print("\n=== REAL vs FAKE Wavelet Contribution ===")

        print("\nREAL:")
        for k, v in real_mean.items():
            print(f"{k}: {v:.6f}")

        print("\nFAKE:")
        for k, v in fake_mean.items():
            print(f"{k}: {v:.6f}")

        bands = ["LL", "LH", "HL", "HH"]
        real_vals = [real_mean[b] for b in bands]
        fake_vals = [fake_mean[b] for b in bands]

        x = np.arange(len(bands))
        width = 0.35

        plt.figure(figsize=(7, 5))
        plt.bar(x - width / 2, real_vals, width, label='Real')
        plt.bar(x + width / 2, fake_vals, width, label='Fake')

        plt.xlabel("Wavelet Bands")
        plt.ylabel("Contribution")
        plt.title("Wavelet Band Contribution: Real vs Fake")
        plt.xticks(x, bands)
        plt.legend()

        save_path = os.path.join(save_dir, "wavelet_real_vs_fake.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        with open(os.path.join(save_dir, "wavelet_real_vs_fake.txt"), "w") as f:
            f.write("REAL:\n")
            for k, v in real_mean.items():
                f.write(f"{k}: {v:.6f}\n")

            f.write("\nFAKE:\n")
            for k, v in fake_mean.items():
                f.write(f"{k}: {v:.6f}\n")

    num_tests = test_loader.dataset.labels.size(0)

    labels = np.vstack(labels).reshape(num_tests)
    predicts = np.vstack(predicts).reshape(num_tests)

    print('\33[91mTest set: {}\n\33[0m'.format(logger_test_name))

    acc = np.sum(labels == predicts) / float(num_tests)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(acc))

    return acc

def main(test_loaders, model):
    print('\nparsed options:\n{}\n'.format(vars(args)))
    acc_list = []
    if args.cuda:
        model.cuda()

    if not args.leave_one_out:
        csv_file = csv.writer(open('{}/{}.csv'.format(args.result_dir, suffix), 'w'), delimiter=',')
        csv_file.writerow(dataset_names) 
    else:
        result_dict = OrderedDict()
        try:
            read_result_dict = load_csv('{}/leave_one_out.csv'.format(args.result_dir),',')
            read_result_dict = read_result_dict.to_dict()
            for key in read_result_dict:
                result_dict[key] = read_result_dict[key][0]
        except Exception as e:
            print(str(e))
        if result_dict is None:
            result_dict = OrderedDict()

    start = args.start_epoch
    end = start + args.epochs
    for test_loader in test_loaders:
        acc = test(test_loader['dataloader'], model, 0, test_loader['name'])*100
        acc_list.append(str(acc))
        if args.leave_one_out:
            result_dict[test_loader['name']] = acc
    
    #write csv file
    if not args.leave_one_out:
        csv_file.writerow(acc_list) 
    else:
        csv_file = csv.writer(open('{}/leave_one_out.csv'.format(args.result_dir), 'w'), delimiter=',')
        name_list = []
        acc_list = []
        for key in result_dict:
            name_list.append(key)
            acc_list.append(result_dict[key])
        csv_file.writerow(name_list)  
        csv_file.writerow(acc_list) 
    
        
if __name__ == '__main__':
    if args.model == 'resnet':
        if args.feature == 'wavelet':

            model = models.resnet34(pretrained=False)

            # Modify first conv layer to accept 12 channels
            original_conv = model.conv1
            model.conv1 = nn.Conv2d(
                in_channels=12,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Copy pretrained weights intelligently
            with torch.no_grad():
                model.conv1.weight[:, :3] = original_conv.weight
                for i in range(3, 12):
                    model.conv1.weight[:, i] = original_conv.weight[:, i % 3]

            model.fc = nn.Linear(model.fc.in_features, 2)

        else:
            model = models.resnet34(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
    elif args.model == 'pggan':
        model = pggan_dnet.SimpleDiscriminator(3, label_size=1, mbstat_avg='all', 
                resolution=256, fmap_max=128, fmap_base=2048, sigmoid_at_end=False)
    elif args.model == 'densenet':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)

    print('{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.epochs))
    load_model = torch.load('{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.epochs), map_location=torch.device('cpu'))
    #load_model = torch.load('{}{}/checkpoint_{}.pth'.format(args.model_dir,suffix,args.epochs))
    model.load_state_dict(load_model['state_dict'])

    test_loaders = create_loaders()
    main(test_loaders, model)
