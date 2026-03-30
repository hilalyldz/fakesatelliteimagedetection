import cv2
import numpy as np

def is_exact_match(img1_path, img2_path):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None:
        raise ValueError("One or both images couldn't be loaded.")
    return np.array_equal(img1, img2)

# Usage
same = is_exact_match(r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\original\115.png", r"C:\Users\yild_hi\Desktop\Datasets\Real_RSCM_224\Real_RSCM_224\image\115.png")
print("Exact Match:", same)
