import os
import cv2
import numpy as np

def is_exact_image_match(path1, path2):
    """Checks if two images are bitwise identical using OpenCV."""
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        return False
    return np.array_equal(img1, img2)

def remove_duplicate_images(fake_dir, real_dir):
    fake_dir = os.path.abspath(fake_dir)
    real_dir = os.path.abspath(real_dir)

    removed = 0
    for filename in os.listdir(fake_dir):
        fake_path = os.path.join(fake_dir, filename)
        real_path = os.path.join(real_dir, filename)

        if os.path.isfile(fake_path) and os.path.isfile(real_path):
            try:
                if is_exact_image_match(fake_path, real_path):
                    os.remove(fake_path)
                    removed += 1
                    print(f"Deleted duplicate: {filename}")
            except Exception as e:
                print(f"Error comparing {filename}: {e}")

    print(f"\nDone. Removed {removed} exact duplicates from 'fake/'.")

# Example usage
fake_directory = r"C:\Users\yild_hi\Desktop\All_Data\fake\satellite"
real_directory = r"C:\Users\yild_hi\Desktop\All_Data\real\satellite"

remove_duplicate_images(fake_directory, real_directory)

