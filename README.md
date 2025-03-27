# DETECTION OF FAKE REMOTE SENSING IMAGES
This project are taken from https://github.com/ColumbiaDVMM/AutoGAN.git
## Dataset
- Dataset DM-AER is taken from https://drive.google.com/drive/folders/1h65vVQvfYzMsmofxTTEIOVSIWdi7zjTo.  DM-AER includes
120,000 real images from different locations around the world, These images includes different weathers and lighting conditions with differing resolutions[2]. Additionally, There are 1,000,000 synthetic
images generated using StyleGAN2 model, mirroring the characteristics of the real images and trained
on them.
- Both of real and fake images have .jpg extension.

## Execution of the Program
```bash
# From project directory
python run_training.py --dataset=CycleGAN --feature=fft --gpu-id=0
```
