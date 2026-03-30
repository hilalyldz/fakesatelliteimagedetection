import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====== CONFIG ======
input_folder = r"C:\Users\yild_hi\Desktop\CMF_Data\fake\satellite\test"   # kendi klasörünü yaz
output_folder = os.path.join(input_folder, "fft_results")

# output klasör oluştur
os.makedirs(output_folder, exist_ok=True)

# desteklenen formatlar
valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def compute_fft(image):
    # grayscale çevir
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # FFT
    f = np.fft.fft2(gray)
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

    fft_img = compute_fft(image)

    # normalize FFT for visualization
    fft_norm = cv2.normalize(fft_img, None, 0, 255, cv2.NORM_MINMAX)
    fft_norm = fft_norm.astype(np.uint8)

    # ====== PLOT ======
    plt.figure(figsize=(8, 4))

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    # FFT
    plt.subplot(1, 2, 2)
    plt.imshow(fft_norm, cmap='gray')
    plt.title("FFT")
    plt.axis("off")

    # save
    save_path = os.path.join(output_folder, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

print("Done! FFT images saved in:", output_folder)