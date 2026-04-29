import matplotlib.pyplot as plt
import seaborn as sns
import os

# Data counts
classes = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']
counts = [1087, 2189, 2386, 13158, 2577]

# Set visual style
plt.style.use('dark_background')
plt.figure(figsize=(10, 6))
colors = sns.color_palette("viridis", len(classes))

# Create bar chart
sns.barplot(x=classes, y=counts, palette=colors)

# Labels
plt.title('Cassava Leaf Disease: Class Distribution', fontsize=16, pad=20, color='#aed581')
plt.xlabel('Disease Type', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)
plt.grid(axis='y', alpha=0.3)

# Add count labels on top of bars
for i, count in enumerate(counts):
    plt.text(i, count + 100, str(count), ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('d:/Kaggle/scratch/class_distribution.png')
print("Saved distribution plot to d:/Kaggle/scratch/class_distribution.png")
