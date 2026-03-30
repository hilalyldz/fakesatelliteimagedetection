import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load color image
image_path = r'C:\Users\yild_hi\PycharmProjects\fakesatelliteimagedetection1\datasets\fake\satellite\test\70.png'  # Replace with your image path
img = Image.open(image_path).convert('RGB')  # Ensure it's in RGB
img_np = np.array(img)

# Split into R, G, B channels
r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]

# Function to compute magnitude spectrum of a channel
def compute_fft(channel):
    f = np.fft.fft2(channel)
    fshift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(fshift) + 1e-8)
    return magnitude

# Compute frequency domain for each channel
r_mag = compute_fft(r)
g_mag = compute_fft(g)
b_mag = compute_fft(b)

# Stack back to form RGB frequency spectrum image
freq_rgb = np.stack([r_mag, g_mag, b_mag], axis=-1)
# Normalize for visualization
freq_rgb = (freq_rgb - freq_rgb.min()) / (freq_rgb.max() - freq_rgb.min()) * 255
freq_rgb = freq_rgb.astype(np.uint8)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_np)
plt.title('Original Image (Spatial Domain)')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(freq_rgb)
plt.title('Frequency Domain (Magnitude Spectrum)')
plt.axis('off')

plt.tight_layout()
plt.show()

