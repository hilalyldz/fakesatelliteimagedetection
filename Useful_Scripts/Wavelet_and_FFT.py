import cv2
import numpy as np
import pywt
import matplotlib.pyplot as plt

# ===== 1. Load Image in COLOR =====
image_path = r"C:\Users\yild_hi\PycharmProjects\fakesatelliteimagedetection1\datasets\fake\satellite\test\000000811.jpg"
img = cv2.imread(image_path)  # Load in BGR
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

if img is None:
    raise ValueError(f"Image not found at {image_path}")

# ===== Helper function: Normalize to 0–255 =====
def normalize(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    return 255 * (arr - arr_min) / (arr_max - arr_min)

# ===== 2. FFT Transformation for each channel =====
fft_channels = []
for i in range(3):  # R, G, B channels
    f = np.fft.fft2(img[:, :, i])
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = normalize(magnitude_spectrum)
    fft_channels.append(magnitude_spectrum.astype(np.uint8))
fft_result = cv2.merge(fft_channels)

# ===== 3. Wavelet Transformation for each channel =====
wavelet_channels = []
for i in range(3):
    coeffs2 = pywt.dwt2(img[:, :, i], 'haar')
    LL, (LH, HL, HH) = coeffs2

    LL_norm = normalize(LL)
    LH_norm = normalize(LH)
    HL_norm = normalize(HL)
    HH_norm = normalize(HH)

    # Combine the wavelet components
    wavelet_img = np.vstack((
        np.hstack((LL_norm, LH_norm)),
        np.hstack((HL_norm, HH_norm))
    ))

    # Resize to original image size
    wavelet_img = cv2.resize(wavelet_img, (img.shape[1], img.shape[0]))
    wavelet_channels.append(wavelet_img.astype(np.uint8))
wavelet_result = cv2.merge(wavelet_channels)

# ===== 4. Visualization =====
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(fft_result)
axes[1].set_title('FFT Magnitude Spectrum')
axes[1].axis('off')

axes[2].imshow(wavelet_result)
axes[2].set_title('Wavelet Transformation')
axes[2].axis('off')

plt.tight_layout()

# ===== 5. Save combined result =====
output_path = r"C:\Users\yild_hi\PycharmProjects\fakesatelliteimagedetection1\datasets_CMF\analysis\fft_wavelet_result_color.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Result saved as {output_path}")
