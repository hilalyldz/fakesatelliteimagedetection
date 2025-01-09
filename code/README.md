# DETECTION OF FAKE REMOTE SENSING IMAGES
## Dataset
- Dataset is taken from https://figshare.com/s/eeedcd150e759ef4353c (Zhao et al, 2021). It consists of real and fake satellite images. 
Real images are from Tocamo. Fake images are generated using CycleGAN. Fake images for city of Tocamo generated using satellite images of Seattle.
Real and fake images consist of 2016 images in total, 1600 for train and 416 for test, separately.
- While real images have .png extension, fake images have .jpg extension.

## Execution of the Program
```bash
# From project directory
python run_training.py --dataset=CycleGAN --feature=fft --gpu-id=0
```
