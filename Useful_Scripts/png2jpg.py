import os
from PIL import Image

# Klasör yolu
folder_path = r"C:\Users\yild_hi\Desktop\GAN_Data\real\satellite\test"

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".png"):
        png_path = os.path.join(folder_path, filename)

        # jpg dosya ismi
        jpg_filename = os.path.splitext(filename)[0] + ".jpg"
        jpg_path = os.path.join(folder_path, jpg_filename)

        # PNG -> JPG çevir
        with Image.open(png_path) as img:
            rgb_img = img.convert("RGB")
            rgb_img.save(jpg_path, "JPEG")

        # PNG dosyasını sil
        os.remove(png_path)

        print(f"{filename} -> {jpg_filename} dönüştürüldü ve PNG silindi.")

print("Tüm işlemler tamamlandı.")