import pandas as pd
import matplotlib.pyplot as plt


file_path = "/Users/nishtman/Desktop/datasets1/monk_datasets/lfw_plot_Annotation.csv"


data = pd.read_csv(file_path)


expected_columns = {"Image Name", "count", "Monk_class"}
if not expected_columns.issubset(data.columns):
    raise ValueError(f"Expected columns {expected_columns}, but got {data.columns}")


data["Monk_class"] = pd.to_numeric(data["Monk_class"], errors="coerce")
data = data.dropna(subset=["Monk_class"])
data["Monk_class"] = data["Monk_class"].astype(int)


class_counts = data.groupby("Monk_class")["count"].sum()


monk_rgb_tones = {
    1: (239, 208, 207), 2: (234, 192, 165), 3: (221, 178, 136),
    4: (200, 150, 108), 5: (184, 124, 93), 6: (161, 96, 67),
    7: (142, 74, 50), 8: (120, 54, 38), 9: (94, 38, 24), 10: (69, 28, 16)
}
skin_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in monk_rgb_tones.values()]

plt.figure(figsize=(12, 6))
bars = plt.bar(class_counts.index, class_counts.values, color=skin_colors, edgecolor='black')


for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
             f"{count}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.xlabel("Monk Skin Tone (MST) Scale", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title("Monk Skin Tone Class Distribution for lfw_Annotation", fontsize=14, fontweight='bold')
plt.xticks(range(1, 11))
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
