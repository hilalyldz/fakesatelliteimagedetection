import cv2
import numpy as np
import matplotlib.pyplot as plt

# === INPUT ===
image_path = r"C:\Users\yild_hi\Desktop\Datasets\FSI_Dataset\fake\Seattle_style\10455_22992.jpg"   # kendi image path'in
output_path = r"C:\Users\yild_hi\PycharmProjects\fakesatelliteimagedetection1\CMF_Image_Analysing\fft_result.png"

# === 1. Görüntüyü oku ===
img = cv2.imread(image_path)

# BGR -> RGB (matplotlib için)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === 2. Grayscale'e çevir (FFT için genelde daha stabil) ===
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# === 3. FFT uygula ===
f = np.fft.fft2(gray)
fshift = np.fft.fftshift(f)

# === 4. Magnitude spectrum hesapla ===
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)  # +1 log(0) hatasını önler

# === 5. Plot ===
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis('off')

# FFT image
plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title("FFT Magnitude Spectrum")
plt.axis('off')

# === 6. Save ===
plt.tight_layout()
plt.savefig(output_path, dpi=300)
plt.close()

print(f"Saved to: {output_path}")