import os


def delete_corresponding_images(folder1, folder2):
    if not os.path.exists(folder1) or not os.path.exists(folder2):
        print("One or both folders do not exist.")
        return

    folder1_images = {os.path.splitext(f)[0] for f in os.listdir(folder1) if f.endswith('.jpg')}
    folder2_images = {f for f in os.listdir(folder2) if f.endswith('.png')}

    for image in folder2_images:
        image_name, ext = os.path.splitext(image)
        if image_name not in folder1_images:
            image_path = os.path.join(folder2, image)
            if os.path.isfile(image_path):
                os.remove(image_path)
                print(f"Deleted: {image_path}")


folder1 = r"C:\\Users\\yild_hi\\PycharmProjects\\fakesatelliteimagedetection1\\datasets\\fake\\satellite\\trainA"
folder2 = r"C:\\Users\\yild_hi\\PycharmProjects\\fakesatelliteimagedetection1\\datasets\\real\\satellite\\trainA"

delete_corresponding_images(folder1, folder2)

