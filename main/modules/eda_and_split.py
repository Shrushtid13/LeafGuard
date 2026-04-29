import os
import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

def perform_eda_and_split(source_dir, output_dir, train_ratio=0.8, val_ratio=0.1):
    print("--- Step 1: Exploratory Data Analysis ---")
    
    classes = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    stats = []
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        count = len([f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))])
        stats.append({'Class': cls, 'Count': count})
    
    df = pd.DataFrame(stats)
    df['Percentage'] = (df['Count'] / df['Count'].sum() * 100).round(2)
    print("\nDataset Distribution:")
    print(df.to_string(index=False))

    # Visualization
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.barplot(data=df, x='Class', y='Count', palette='viridis', hue='Class', legend=False)
    plt.title('Cassava Leaf Disease: Initial Class Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(output_dir), 'eda_distribution.png')
    plt.savefig(plot_path)
    print(f"\nEDA Chart saved to: {plot_path}")

    print("\n--- Step 2: Systematic Data Splitting ---")
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        images = [f for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        random.seed(42)
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
            for img in tqdm(split_images, desc=f"Moving {cls} -> {split}", leave=False):
                shutil.copy(os.path.join(cls_path, img), os.path.join(split_path, img))
    
    print(f"Data splitting complete. Folders created in: {output_dir}")

if __name__ == "__main__":
    source = r"d:\Kaggle\cassava_leaf_disease_classification\data"
    output = r"d:\Kaggle\cassava_leaf_disease_classification\processed"
    perform_eda_and_split(source, output)
