import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
input_folder = r"C:\Users\yild_hi\Desktop\FSI_Dmaer_Dataset\real\satellite\test"  # kendi klasörünü yaz
output_folder = os.path.join(input_folder, "fft_results")

# output klasör oluştur
os.makedirs(output_folder, exist_ok=True)

# desteklenen formatlar
valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


# ====== FUNCTIONS ======

def apply_high_pass_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # low-frequency (blur)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)

    # high-pass = original - blur
    high_pass = cv2.subtract(gray, blur)

    return high_pass


def compute_fft(image):
    # FFT
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)

    # magnitude spectrum
    magnitude = np.log(np.abs(fshift) + 1)

    return magnitude


# ====== PROCESS ======
for filename in os.listdir(input_folder):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)

    if image is None:
        continue

    # 1. High-pass filter
    high_pass_img = apply_high_pass_filter(image)

    # 2. FFT
    fft_img = compute_fft(high_pass_img)

    # normalize FFT for visualization
    fft_norm = cv2.normalize(fft_img, None, 0, 255, cv2.NORM_MINMAX)
    fft_norm = fft_norm.astype(np.uint8)

    # ====== PLOT ======
    plt.figure(figsize=(12, 4))

    # original image
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # high-pass image
    plt.subplot(1, 3, 2)
    plt.imshow(high_pass_img, cmap='gray')
    plt.title("High-Pass")
    plt.axis("off")

    # FFT result
    plt.subplot(1, 3, 3)
    plt.imshow(fft_norm, cmap='gray')
    plt.title("FFT (After High-Pass)")
    plt.axis("off")

    # save
    save_path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

print("Done! FFT images saved in:", output_folder)