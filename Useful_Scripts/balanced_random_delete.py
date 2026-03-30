import os
import random

def balanced_random_delete(folder_path, delete_count, seed=42):
    random.seed(seed)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    # Resimleri topla
    images = [os.path.join(folder_path, f)
              for f in os.listdir(folder_path)
              if f.lower().endswith(image_extensions)]

    total_images = len(images)

    if delete_count > total_images:
        print("Silinecek sayı toplam resimden büyük!")
        return

    print(f"Toplam resim: {total_images}")
    print(f"Silinecek resim sayısı: {delete_count}")

    # Alfabetik sırala (denge için)
    images.sort()

    # Listeyi segmentlere böl
    segments = delete_count
    segment_size = total_images // segments

    selected_for_delete = []

    for i in range(segments):
        start = i * segment_size
        end = start + segment_size

        if i == segments - 1:
            end = total_images

        segment = images[start:end]
        if segment:
            chosen = random.choice(segment)
            selected_for_delete.append(chosen)

    # Silme işlemi
    for img_path in selected_for_delete:
        os.remove(img_path)

    print("Silme işlemi tamamlandı ✔")


# Kullanım
folder = r"C:\Users\yild_hi\Desktop\cmfdata\fake"
balanced_random_delete(folder, delete_count=1955)
