import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


files = {
    "ffhq Dataset": "/Users/nishtman/Desktop/GPU_output/ffhq/Monk/predicted_images_color_metrics.csv",
    "lfw Dataset": "/Users/nishtman/Desktop/GPU_output/lfw/Monk/predicted_images_color_metrics.csv",
    "ffhq StyleGAN": "/Users/nishtman/Desktop/GPU_output/ffhq_Style_GAN/Monk/predicted_images_color_metrics.csv",
    "lfw Hyperparameter": "/Users/nishtman/Desktop/GPU_output/lfw_Hyperparameter/Monk/predicted_images_color_metrics.csv"
}


color_spaces = {
    "HSV": "Closest Monk Tone HSV",
    "YCbCr": "Closest Monk Tone YCbCr",
    "Avg Grayscale": "Closest Monk Tone Average Grayscale",
    "Weighted Grayscale": "Closest Monk Tone Weighted Grayscale"
}


color_classes = [
    [246, 237, 228], [243, 231, 219], [247, 234, 208], [234, 218, 186],
    [215, 189, 150], [160, 126, 86], [130, 92, 67], [96, 65, 52], 
    [58, 49, 42], [41, 36, 32]
]
skin_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in color_classes]


bias_scores = {}


ideal_distribution = np.full(10, 0.1)


for space, column in color_spaces.items():
    plt.figure(figsize=(20, 5))  
    
    for idx, (dataset_name, file_path) in enumerate(files.items()):
        df = pd.read_csv(file_path)

        if column in df.columns:
            
            class_counts = df[column].value_counts(normalize=True).sort_index()
            
           
            class_distribution = np.zeros(10)
            for i in range(1, 11):  
                if i in class_counts:
                    class_distribution[i-1] = class_counts[i]
            
            
            bias_score = np.sum(np.abs(class_distribution - ideal_distribution))
            bias_scores[f"{dataset_name} - {space}"] = bias_score
            
         
            plt.subplot(1, 4, idx + 1)  
            bars = plt.bar(np.arange(1, 11), class_distribution, color=skin_colors)
            plt.axhline(y=0.1, color='black', linestyle='--', label='Ideal 10%')
            plt.title(f"{dataset_name} (predicted images)", fontsize=12)
            plt.xlabel("Monk Skin Tone Class", fontsize=10)
            plt.xticks(range(1, 11))
            plt.ylim(0, max(class_distribution) + 0.05)
            plt.grid(axis='y', linestyle='--', alpha=0.5)

    
            for bar, count in zip(bars, class_distribution):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                         f"{count:.2f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.suptitle(f"Evaluation of Bias Across 4 Datasets in {space} Color Space for predicted_images", 
                 fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()


bias_df = pd.DataFrame(list(bias_scores.items()), columns=["Model", "Bias Score"])
bias_df = bias_df.sort_values(by="Bias Score", ascending=False)


plt.figure(figsize=(12, 8))
sns.barplot(y=bias_df["Model"], x=bias_df["Bias Score"], palette="viridis")
plt.xlabel("Bias Score (Lower is Better)", fontsize=12)
plt.ylabel("Color Spaces", fontsize=10)
plt.title("Evaluation of Bias Across Color Spaces and Datasets for predicted_images", fontsize=14)
plt.yticks(fontsize=10, rotation=0)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout() 
plt.show()


print("\nFinal Bias Scores:")
print(bias_df.to_string(index=False))
