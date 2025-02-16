import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


monk_gray_tones_weighted = {
    1: 216.38,
    2: 205.71,
    3: 187.63,
    4: 167.02,
    5: 145.77,
    6: 121.75,
    7: 101.36,
    8: 82.74,
    9: 63.76,
    10: 46.31
}

monk_gray_tones_average = {
    1: 218.00,
    2: 197.00,
    3: 178.33,
    4: 152.67,
    5: 133.67,
    6: 108.67,
    7: 88.67,
    8: 70.67,
    9: 52.00,
    10: 37.67
}


monk_colors = {
    1: (0.937, 0.816, 0.812),
    2: (0.918, 0.753, 0.647),
    3: (0.867, 0.698, 0.533),
    4: (0.784, 0.588, 0.424),
    5: (0.722, 0.486, 0.365),
    6: (0.631, 0.376, 0.263),
    7: (0.557, 0.290, 0.196),
    8: (0.471, 0.212, 0.149),
    9: (0.369, 0.149, 0.094),
    10: (0.271, 0.110, 0.063)
}



def gray_to_rgb(gray_value):
    normalized = gray_value / 255.0
    return (normalized, normalized, normalized)


files = {
    "ffhq Dataset": "/Users/nishtman/Desktop/GPU_output/ffhq/Monk/predicted_images_color_metrics.csv",
    "lfw Dataset": "/Users/nishtman/Desktop/GPU_output/lfw/Monk/predicted_images_color_metrics.csv",
    "ffhq StyleGAN": "/Users/nishtman/Desktop/GPU_output/ffhq_Style_GAN/Monk/predicted_images_color_metrics.csv",
    "lfw Hyperparameter": "/Users/nishtman/Desktop/GPU_output/lfw_Hyperparameter/Monk/predicted_images_color_metrics.csv"
}


output_directory = "/Users/nishtman/Desktop/final/"
os.makedirs(output_directory, exist_ok=True)


for label, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue

    try:
        data = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading {label}: {e}")
        continue


    required_columns = ['Image Name', 'Closest Monk Tone Average Grayscale', 'Closest Monk Tone Weighted Grayscale']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        print(f"Required columns {missing_columns} not found in {label}: {path}")
        continue

    
    frequency_weighted = data['Closest Monk Tone Weighted Grayscale'].value_counts().sort_index()
    frequency_average = data['Closest Monk Tone Average Grayscale'].value_counts().sort_index()

 
    for i in range(1, 11):
        if i not in frequency_weighted:
            frequency_weighted[i] = 0
        if i not in frequency_average:
            frequency_average[i] = 0
    frequency_weighted = frequency_weighted.sort_index()
    frequency_average = frequency_average.sort_index()

    
    plt.figure(figsize=(12, 6))
    bar_width = 0.4
    for i, (freq, color, gray_value) in enumerate(zip(
        frequency_weighted.values,
        [monk_colors[k] for k in frequency_weighted.index],
        [gray_to_rgb(monk_gray_tones_weighted[k]) for k in frequency_weighted.index]
    )):
        plt.bar(i - bar_width / 2, freq, color=gray_value, width=bar_width, edgecolor='none', label='Gray' if i == 0 else "")
        plt.bar(i + bar_width / 2, freq, color=color, width=bar_width, edgecolor='none', label='Monk Tone' if i == 0 else "")
        plt.text(i, freq, f'{int(freq)}', ha='center', va='bottom')
    plt.title(f'{label} - Frequency of Monk Tone Classes Based on Weighted Grayscale')
    plt.xlabel('Monk Tone Class')
    plt.ylabel('Frequency')
    plt.xticks(range(10), range(1, 11))
    plt.legend(loc='upper right')
    plt.show()

   
    plt.figure(figsize=(12, 6))
    for i, (freq, color, gray_value) in enumerate(zip(
        frequency_average.values,
        [monk_colors[k] for k in frequency_average.index],
        [gray_to_rgb(monk_gray_tones_average[k]) for k in frequency_average.index]
    )):
        plt.bar(i - bar_width / 2, freq, color=gray_value, width=bar_width, edgecolor='none', label='Gray' if i == 0 else "")
        plt.bar(i + bar_width / 2, freq, color=color, width=bar_width, edgecolor='none', label='Monk Tone' if i == 0 else "")
        plt.text(i, freq, f'{int(freq)}', ha='center', va='bottom')
    plt.title(f'{label} - Frequency of Monk Tone Classes Based on Average Grayscale')
    plt.xlabel('Monk Tone Class')
    plt.ylabel('Frequency')
    plt.xticks(range(10), range(1, 11))
    plt.legend(loc='upper right')
    plt.show()

print(f"Files processed and saved in {output_directory}")
