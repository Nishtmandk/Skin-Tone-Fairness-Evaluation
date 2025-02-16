import os
import pandas as pd
import colorsys
import numpy as np
import matplotlib.pyplot as plt

# Define Monk Skin Tone RGB values
monk_rgb_tones = {
    1: (239, 208, 207),
    2: (234, 192, 165),
    3: (221, 178, 136),
    4: (200, 150, 108),
    5: (184, 124, 93),
    6: (161, 96, 67),
    7: (142, 74, 50),
    8: (120, 54, 38),
    9: (94, 38, 24),
    10: (69, 28, 16)
}


monk_hsv = {k: colorsys.rgb_to_hsv(v[0] / 255, v[1] / 255, v[2] / 255) for k, v in monk_rgb_tones.items()}
monk_ycbcr = {k: (
    16 + (65.738 * v[0] + 129.057 * v[1] + 25.064 * v[2]) / 256,
    128 - (37.945 * v[0] + 74.494 * v[1] - 112.439 * v[2]) / 256,
    128 + (112.439 * v[0] - 94.154 * v[1] - 18.285 * v[2]) / 256
) for k, v in monk_rgb_tones.items()}


def rgb_to_hsv(rgb):
    return colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)


def rgb_to_ycbcr(rgb):
    r, g, b = rgb
    y = 16 + (65.738 * r + 129.057 * g + 25.064 * b) / 256
    cb = 128 - (37.945 * r + 74.494 * g - 112.439 * b) / 256
    cr = 128 + (112.439 * r - 94.154 * g - 18.285 * b) / 256
    return y, cb, cr


def closest_class(value, reference):
    distances = {k: np.linalg.norm(np.array(value) - np.array(ref)) for k, ref in reference.items()}
    return min(distances, key=distances.get)


files = {
    "ffhq Dataset":"/Users/nishtman/Desktop/GPU_output/ffhq/Monk/albedo_images_color_metrics.csv",
    "lfw Dataset": "/Users/nishtman/Desktop/GPU_output/lfw/Monk/albedo_images_color_metrics.csv",
    "ffhq StyleGAN": "/Users/nishtman/Desktop/GPU_output/ffhq_Style_GAN/Monk/albedo_images_color_metrics.csv",
    "lfw Hyperparameter": "/Users/nishtman/Desktop/GPU_output/lfw_Hyperparameter/Monk/albedo_images_color_metrics.csv" 
}



dataset_sizes = {
    "ffhq Dataset": 66011,
    "lfw Dataset": 12925,
    "ffhq StyleGAN": 8850,
    "lfw Hyperparameter": 30704
}


frequency_hsv_data = {label: np.zeros(10) for label in files.keys()}
frequency_ycbcr_data = {label: np.zeros(10) for label in files.keys()}


for label, path in files.items():
    if not os.path.exists(path):
        print(f" File not found: {path}")
        continue


    data = pd.read_csv(path)

    
    data['Closest Monk Tone HSV'] = data.apply(
        lambda row: closest_class(rgb_to_hsv((row['Average R'], row['Average G'], row['Average B'])), monk_hsv), axis=1
    )
    data['Closest Monk Tone YCbCr'] = data.apply(
        lambda row: closest_class(rgb_to_ycbcr((row['Average R'], row['Average G'], row['Average B'])), monk_ycbcr), axis=1
    )

    
    frequency_hsv = data['Closest Monk Tone HSV'].value_counts().sort_index()
    frequency_ycbcr = data['Closest Monk Tone YCbCr'].value_counts().sort_index()

  
    dataset_size = dataset_sizes[label]
    for i in range(1, 11):
        frequency_hsv_data[label][i - 1] = round((frequency_hsv.get(i, 0) / dataset_size) * 100, 2)
        frequency_ycbcr_data[label][i - 1] = round((frequency_ycbcr.get(i, 0) / dataset_size) * 100, 2)


print("Monk Skin Tone (MST) Distribution Across Datasets (HSV)")
print("=" * 120)
print(f"{'Dataset':<25} {'MST-1':<8} {'MST-2':<8} {'MST-3':<8} {'MST-4':<8} {'MST-5':<8} {'MST-6':<8} {'MST-7':<8} {'MST-8':<8} {'MST-9':<8} {'MST-10':<8}")
print("-" * 120)

for label in files.keys():
    hsv_values = " ".join(f"{val:<8}" for val in frequency_hsv_data[label])
    print(f"{label:<25} {hsv_values}")

print("-" * 120)
print("Monk Skin Tone (MST) Distribution Across Datasets (YCbCr)")
print("=" * 120)
print(f"{'Dataset':<25} {'MST-1':<8} {'MST-2':<8} {'MST-3':<8} {'MST-4':<8} {'MST-5':<8} {'MST-6':<8} {'MST-7':<8} {'MST-8':<8} {'MST-9':<8} {'MST-10':<8}")
print("-" * 120)

for label in files.keys():
    ycbcr_values = " ".join(f"{val:<8}" for val in frequency_ycbcr_data[label])
    print(f"{label:<25} {ycbcr_values}")

print("=" * 120)




plt.figure(figsize=(12, 6))
bar_width = 0.2
x_positions = np.arange(10)


for i, (label, percentages) in enumerate(frequency_hsv_data.items()):
    plt.bar(x_positions + i * bar_width, percentages, width=bar_width, 
            label=label, edgecolor='black')

plt.xlabel("Monk Skin Tone (MST) Scale", fontsize=12, fontweight="bold")
plt.ylabel("Percentage of Dataset Representing Each Frequency (%)", fontsize=12, fontweight="bold")
plt.title("MST Distribution Based on HSV Mapping for Albedo Images(without lighting)", fontsize=14, fontweight="bold")
plt.xticks(x_positions + bar_width * 1.5, range(1, 11))
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))


for i, (label, percentages) in enumerate(frequency_ycbcr_data.items()):
    plt.bar(x_positions + i * bar_width, percentages, width=bar_width, 
            label=label, edgecolor='black')

plt.xlabel("Monk Skin Tone (MST) Scale", fontsize=12, fontweight="bold")
plt.ylabel("Percentage of Dataset Representing Each Frequency (%)", fontsize=12, fontweight="bold")
plt.title("MST Distribution Based on YCbCr Mapping for Albedo Images(without lighting)", fontsize=14, fontweight="bold")
plt.xticks(x_positions + bar_width * 1.5, range(1, 11))
plt.legend(loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
