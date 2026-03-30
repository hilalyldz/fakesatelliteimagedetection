import os
from PIL import Image

def convert_png_to_jpg_and_delete(root_dir):
    """
    Convert all .png images in root_dir (and subdirectories) to .jpg,
    and delete the original .png file after successful conversion.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.png'):
                png_path = os.path.join(dirpath, filename)
                jpg_path = os.path.splitext(png_path)[0] + '.jpg'

                try:
                    with Image.open(png_path) as img:
                        rgb_img = img.convert('RGB')  # Ensure no alpha channel
                        rgb_img.save(jpg_path, 'JPEG', quality=95)
                    os.remove(png_path)  # Delete original .png
                    print(f"Converted and deleted: {png_path} → {jpg_path}")
                except Exception as e:
                    print(f"Failed to convert {png_path}: {e}")

# Example usage
convert_png_to_jpg_and_delete(
    r"C:\Users\yild_hi\Desktop\CMF_Data\val\fake"
)
