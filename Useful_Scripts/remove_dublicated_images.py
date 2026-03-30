import os
import hashlib

def calculate_hash(file_path, chunk_size=8192):
    """Dosyanın SHA-256 hash değerini hesaplar."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def remove_duplicate_images(folder_path):
    hashes = {}
    deleted_files = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)

            # Sadece resim uzantıları
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    file_hash = calculate_hash(file_path)

                    if file_hash in hashes:
                        print(f"Siliniyor: {file_path}")
                        os.remove(file_path)
                        deleted_files.append(file_path)
                    else:
                        hashes[file_hash] = file_path

                except Exception as e:
                    print(f"Hata oluştu: {file_path} - {e}")

    print("\nToplam silinen dosya sayısı:", len(deleted_files))


# Kullanım
folder = r"C:\Users\yild_hi\Desktop\cmfdata\real"
remove_duplicate_images(folder)
