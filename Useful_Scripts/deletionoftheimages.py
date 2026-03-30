import os
import random


# Set your directory path
directory = "C:/Users/yild_hi/PycharmProjects/fakesatelliteimagedetection1/datasets_50/real/satellite/trainA"

# Get the list of files in the directory
files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

# Check if there are at least 800 files
if len(files) < 800:
    print("Not enough files to delete (less than 800 found).")
else:
    # Select 800 random files to delete
    files_to_delete = random.sample(files, 800)

    # Delete selected files
    for file in files_to_delete:
        file_path = os.path.join(directory, file)
        try:
            os.remove(file_path)
            print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

print("Deletion process completed.")
