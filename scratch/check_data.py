import os
from PIL import Image
from tqdm import tqdm

def check_images(data_dir):
    corrupt_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in tqdm(files, desc=f"Checking {os.path.basename(root)}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"Corrupt image: {file_path} - {e}")
                    corrupt_files.append(file_path)
    return corrupt_files

if __name__ == "__main__":
    data_dir = r"d:\Kaggle\cassava_leaf_disease_classification\data"
    corrupt = check_images(data_dir)
    print(f"Total corrupt images: {len(corrupt)}")
    if corrupt:
        print("Corrupt files list:")
        for f in corrupt:
            print(f)
