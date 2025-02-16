import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


monk_gray_tones = {
    "Average": {
        1: 218.00, 2: 197.00, 3: 178.33, 4: 152.67, 5: 133.67,
        6: 108.67, 7: 88.67, 8: 70.67, 9: 52.00, 10: 37.67
    },
    "Weighted": {
        1: 211.35, 2: 192.73, 3: 176.55, 4: 153.48, 5: 135.55,
        6: 112.84, 7: 93.66, 8: 77.46, 9: 59.73, 10: 46.22
    }
}


dataset_sizes = {
    "ffhq Dataset": 66011,
    "lfw Dataset": 12925,
    "ffhq StyleGAN": 8850,
    "lfw Hyperparameter": 30704
}


dataset_colors = {
    "ffhq Dataset": "#77dd77",
    "lfw Dataset": "#ff6961",
    "ffhq StyleGAN": "#76b5c5",
    "lfw Hyperparameter": "#ffb347"
}


files = {
    "ffhq Dataset": "/Users/nishtman/Desktop/GPU_output/ffhq/Monk/albedo_images_color_metrics.csv",
    "lfw Dataset": "/Users/nishtman/Desktop/GPU_output/lfw/Monk/albedo_images_color_metrics.csv",
    "ffhq StyleGAN": "/Users/nishtman/Desktop/GPU_output/ffhq_Style_GAN/Monk/albedo_images_color_metrics.csv",
    "lfw Hyperparameter": "/Users/nishtman/Desktop/GPU_output/lfw_Hyperparameter/Monk/albedo_images_color_metrics.csv"
}


frequency_data = {
    "Average": {label: np.zeros(10) for label in files.keys()},
    "Weighted": {label: np.zeros(10) for label in files.keys()}
}


for label, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"⚠️ Error reading {label}: {e}")
        continue

  
    if 'Closest Monk Tone Average Grayscale' not in data.columns or 'Closest Monk Tone Weighted Grayscale' not in data.columns:
        print(f" Required column missing in {label}: {path}")
        continue

    
    for method in ["Average", "Weighted"]:
        column_name = f'Closest Monk Tone {method} Grayscale'
        frequency = data[column_name].value_counts().sort_index()

        
        dataset_size = dataset_sizes[label]
        for i in range(1, 11):
            frequency_data[method][label][i - 1] = round((frequency.get(i, 0) / dataset_size) * 100, 2)


for method in ["Average", "Weighted"]:
    print(f" Monk Skin Tone (MST) Distribution Across Datasets ({method} Grayscale)")
    print("=" * 120)
    print(f"{'Dataset':<25} {'MST-1':<8} {'MST-2':<8} {'MST-3':<8} {'MST-4':<8} {'MST-5':<8} {'MST-6':<8} {'MST-7':<8} {'MST-8':<8} {'MST-9':<8} {'MST-10':<8}")
    print("-" * 120)

    for label in files.keys():
        values = " ".join(f"{val:<8}" for val in frequency_data[method][label])
        print(f"{label:<25} {values}")

    print("=" * 120)


plt.figure(figsize=(12, 6))
bar_width = 0.2
x_positions = np.arange(10)

for i, (label, percentages) in enumerate(frequency_data["Average"].items()):
    plt.bar(x_positions + i * bar_width, percentages, width=bar_width, 
            color=dataset_colors[label], edgecolor='black', label=label)

plt.xlabel("Monk Skin Tone (MST) Scale", fontsize=12, fontweight="bold")
plt.ylabel("Percentage of Dataset Representing Each Frequency (%)", fontsize=12, fontweight="bold")
plt.title("MST Distribution Based on Average Grayscale Mapping for Albedo Images(without lighting)", fontsize=14, fontweight="bold")
plt.xticks(x_positions + bar_width * 1.5, range(1, 11))
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()




plt.figure(figsize=(12, 6))

for i, (label, percentages) in enumerate(frequency_data["Weighted"].items()):
    plt.bar(x_positions + i * bar_width, percentages, width=bar_width, 
            color=dataset_colors[label], edgecolor='black', label=label)

plt.xlabel("Monk Skin Tone (MST) Scale", fontsize=12, fontweight="bold")
plt.ylabel("Percentage of Dataset Representing Each Frequency (%)", fontsize=12, fontweight="bold")
plt.title("MST Distribution Based on Weighted Grayscale Mapping for Albedo Images(without lighting)", fontsize=14, fontweight="bold")
plt.xticks(x_positions + bar_width * 1.5, range(1, 11))
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
