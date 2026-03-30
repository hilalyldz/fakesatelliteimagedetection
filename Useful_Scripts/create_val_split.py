import os
import random
import shutil


def create_validation_split(source_folder, val_folder, split_ratio=0.2):
    # Validation klasörü yoksa oluştur
    os.makedirs(val_folder, exist_ok=True)

    # Resim dosyalarını topla
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images = [f for f in os.listdir(source_folder)
              if f.lower().endswith(image_extensions)]

    total_images = len(images)
    val_count = int(total_images * split_ratio)

    print(f"Toplam resim sayısı: {total_images}")
    print(f"Validation için seçilecek: {val_count}")

    # Karıştır
    random.shuffle(images)

    # İlk %20'yi al
    val_images = images[:val_count]

    # Kopyala
    for img in val_images:
        src_path = os.path.join(source_folder, img)
        dst_path = os.path.join(val_folder, img)
        shutil.move(src_path, dst_path)

    print("Validation set oluşturuldu ✔")


# Kullanım
source = r"C:\Users\yild_hi\Desktop\cmfdata\real"
validation = r"C:\Users\yild_hi\Desktop\cmfdata\val\real"

create_validation_split(source, validation, split_ratio=0.2)
