import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt

# ===== CONFIG =====
input_folder = r"C:\Users\yild_hi\Desktop\FSI_Dmaer_Dataset\fake\satellite\test"
output_folder = os.path.join(input_folder, "dwt_only")

os.makedirs(output_folder, exist_ok=True)

valid_ext = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

# ===== FUNCTIONS =====

def compute_dwt(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    LL, (LH, HL, HH) = pywt.dwt2(gray, 'haar')
    return LL, LH, HL, HH

def normalize(img):
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def enhance(img):
    return cv2.equalizeHist(img)

# ===== PROCESS =====

for filename in os.listdir(input_folder):
    if not any(filename.lower().endswith(ext) for ext in valid_ext):
        continue

    path = os.path.join(input_folder, filename)
    image = cv2.imread(path)

    if image is None:
        continue

    # DWT
    LL, LH, HL, HH = compute_dwt(image)

    LL = enhance(normalize(LL))
    LH = enhance(normalize(LH))
    HL = enhance(normalize(HL))
    HH = enhance(normalize(HH))

    # ===== PLOT (TEK SATIR) =====
    plt.figure(figsize=(14, 3))

    titles = ["LL (Low Freq)", "LH (Horizontal)", "HL (Vertical)", "HH (Diagonal)"]
    images = [LL, LH, HL, HH]

    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i], fontsize=10)
        plt.axis("off")

    plt.tight_layout()

    # SAVE
    save_path = os.path.join(output_folder, filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

print("Done! DWT-only images saved in:", output_folder)