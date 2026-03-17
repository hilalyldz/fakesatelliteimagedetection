#!/usr/bin/python
#-*- coding: utf-8 -*- 
#===========================================================
#  File Name: run_test.py
#  Author: Xu Zhang, Columbia University
#  Creation Date: 09-07-2019
#  Last Modified: Tue Oct 15 16:39:19 2019
#
#  Usage: python run_test.py -h
#  Description: Evaluate a GAN image detector
#
#  Copyright (C) 2019 Xu Zhang
#  All rights reserved.
# 
#  This file is made available under
#  the terms of the BSD license (see the COPYING file).
#===========================================================
import numpy as np
import scipy.io as sio
import time
import os
import sys
import subprocess
import shlex
import argparse
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch GAN Image Detection')

####################################################################
# Parse command line
####################################################################
parser.add_argument('--dataset', type=str, default='CycleGAN', help='Training dataset select from: CycleGAN and AutoGAN')
parser.add_argument('--feature', default='image', help='Feature used for training, choose from image and fft')

args = parser.parse_args()

gpu_set = ['0']

#Compare image and spectrum
if args.feature == 'image':
    parameter_set = [
            ' --feature=image ',
            ]
elif args.feature == 'fft':
    parameter_set = [
            ' --feature=fft '
            ]
elif args.feature == 'wavelet':
    parameter_set = [
            ' --feature=wavelet '
            ]
#Compare different frequency band
#parameter_set = [
#        ' --feature=fft '
#        ' --feature=fft --mode=1'
#        ' --feature=fft --mode=2'
#        ' --feature=fft --mode=3'
#        ]
else:
    print('Not a valid feature!')
    exit(-1)



number_gpu = len(gpu_set)

# Create timestamped log directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_dir = f"./logs_test/run_test_{timestamp}/"
os.makedirs(log_dir, exist_ok=True)

if args.dataset == 'CycleGAN':
    datasets = ['satellite']
elif args.dataset == 'AutoGAN':
    datasets = ['horse_auto', 'zebra_auto', 'summer_auto', 'winter_auto', 'apple_auto', 'orange_auto', 'facades_auto', 'cityscapes_auto', 'satellite_auto', 'ukiyoe_auto', 'vangogh_auto', 'cezanne_auto', 'monet_auto', 'photo_auto']
else:
    print('Not a valid dataset!')
    exit(-1)

#leave one out setting 
#datasets = ['horse+zebra --leave_one_out ', 'apple+orange --leave_one_out ',
#            'summer+winter --leave_one_out ', 'cityscapes --leave_one_out ', 
#            'satellite --leave_one_out ', 'facades --leave_one_out ', 
#            'fold6 --leave_one_out ', 'fold7 --leave_one_out ',
#            'fold8 --leave_one_out ', 'fold9 --leave_one_out ']
process_set = []

def run_command(command, log_file_path):
    """Run a command and both print output and save to a log file."""
    with open(log_file_path, "w") as log_file:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        for line in process.stdout:
            print(line, end='')     # print to terminal
            log_file.write(line)    # write to log file
        process.wait()

for dataset in datasets:
    for idx, parameter in enumerate(parameter_set):
        print('Test Parameter: {}'.format(parameter))
        command = 'python ./code/GAN_Detection_Test.py --training-set {} --model=resnet --test-set=transposed_conv --data_augment \
                --batch-size=16 --test-batch-size=16 --epochs 20 {}  --gpu-id {} --model-dir ./model_resnet_cmf/ '\
                .format(dataset, parameter, gpu_set[idx%number_gpu]) 
        print(command)

        log_file_path = os.path.join(log_dir, f"{dataset}_{args.feature}_gpu{gpu_set[idx % number_gpu]}.log")

        # Run command and tee output
        run_command(command, log_file_path)


        if (idx+1)%number_gpu == 0:
            print('Wait for process end')
            for sub_process in process_set:
                sub_process.wait()
        
            process_set = []
    
        time.sleep(10)
    
    for sub_process in process_set:
        sub_process.wait()

