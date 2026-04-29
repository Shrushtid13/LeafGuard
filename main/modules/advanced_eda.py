import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from tqdm import tqdm

# Constants
DATA_DIR = r"d:\Kaggle\cassava_leaf_disease_classification\data"
SAVE_DIR = r"d:\Kaggle\main\eda_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

def generate_visualizations():
    print("--- Professional EDA: Generating 5+ Hackathon-Ready Assets ---")
    
    classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    data = []
    
    # 1. Gather Basic Metadata & Stats
    for cls in classes:
        cls_path = os.path.join(DATA_DIR, cls)
        files = os.listdir(cls_path)
        for f in files[:200]: # Sample for speed in stats, but full count for dist
            img_path = os.path.join(cls_path, f)
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    arr = np.array(img)
                    mean_val = np.mean(arr)
                    r_mean, g_mean, b_mean = np.mean(arr, axis=(0,1))
                    data.append({
                        'Class': cls, 'Width': w, 'Height': h, 
                        'Brightness': mean_val, 'R': r_mean, 'G': g_mean, 'B': b_mean
                    })
            except: continue

    df = pd.DataFrame(data)
    
    # --- PLOT 1: Class Distribution ---
    print("Plotting 1/5: Class Distribution...")
    counts = {cls: len(os.listdir(os.path.join(DATA_DIR, cls))) for cls in classes}
    plt.figure(figsize=(10, 6))
    sns.set_palette("viridis")
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), hue=list(counts.keys()), legend=False)
    plt.title('Dataset Balance: Sample Count per Class', fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '01_class_distribution.png'))

    # --- PLOT 2: Sample Grid ---
    print("Plotting 2/5: Representative Samples Grid...")
    fig, axes = plt.subplots(len(classes), 5, figsize=(15, 3*len(classes)))
    for i, cls in enumerate(classes):
        cls_path = os.path.join(DATA_DIR, cls)
        samples = random.sample(os.listdir(cls_path), 5)
        for j, s in enumerate(samples):
            img = Image.open(os.path.join(cls_path, s))
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0: axes[i, j].set_title(cls, fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '02_sample_images_grid.png'))

    # --- PLOT 3: Aspect Ratio/Resolutions ---
    print("Plotting 3/5: Image Resolutions Distribution...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='Width', y='Height', hue='Class', alpha=0.5)
    plt.title('Image Dimensions Consistency Check')
    plt.savefig(os.path.join(SAVE_DIR, '03_image_resolutions.png'))

    # --- PLOT 4: Brightness Distribution ---
    print("Plotting 4/5: Brightness Distribution per Class...")
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='Brightness', hue='Class', fill=True, alpha=0.3)
    plt.title('Luminance Distribution (Field Lighting Analysis)')
    plt.xlabel('Average Pixel Intensity')
    plt.savefig(os.path.join(SAVE_DIR, '04_pixel_intensity_dist.png'))

    # --- PLOT 5: RGB Channel Analysis ---
    print("Plotting 5/5: Average Color Channel Intensity (Spectral Profile)...")
    df_melt = df.melt(id_vars='Class', value_vars=['R', 'G', 'B'], var_name='Channel', value_name='Intensity')
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_melt, x='Class', y='Intensity', hue='Channel', palette=['red', 'green', 'blue'])
    plt.title('Spectral Profile: RGB Intensity Boxplots')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, '05_average_color_channels.png'))

    print(f"\n✅ All 5 professional EDA plots saved to: {SAVE_DIR}")

if __name__ == "__main__":
    generate_visualizations()
