import os
import shutil
import random
from tqdm import tqdm

def split_data(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.shuffle(images)
        
        n_total = len(images)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        splits = {
            'train': images[:n_train],
            'val': images[n_train:n_train+n_val],
            'test': images[n_train+n_val:]
        }
        
        for split, split_images in splits.items():
            split_path = os.path.join(output_dir, split, cls)
            os.makedirs(split_path, exist_ok=True)
            for img in tqdm(split_images, desc=f"Moving {cls} to {split}"):
                shutil.copy(os.path.join(cls_path, img), os.path.join(split_path, img))

if __name__ == "__main__":
    source = r"d:\Kaggle\cassava_leaf_disease_classification\data"
    output = r"d:\Kaggle\cassava_leaf_disease_classification\processed"
    split_data(source, output)
    print("Data splitting complete.")
