import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  datasets
ffhq_data = pd.read_csv('/Users/nishtman/Desktop/GPU_output/OFIQ_CSV_scalar/scaled_ffhq_albedo_images.csv')
lfw_data = pd.read_csv('/Users/nishtman/Desktop/GPU_output/OFIQ_CSV_scalar/scaled_lfw_albedo_images.csv')
stylegan_data = pd.read_csv('/Users/nishtman/Desktop/GPU_output/OFIQ_CSV_scalar/scaled_ffhq_Style_GAN_albedo_images.csv')
lfw_hyperparameter_data = pd.read_csv('/Users/nishtman/Desktop/GPU_output/OFIQ_CSV_scalar/scaled_lfw_predicted_images.csv')

# Feature list 
features = [
    'Sharpness.scalar',
    'IlluminationUniformity.scalar',
    'LuminanceMean.scalar',
    'LuminanceVariance.scalar',
    'DynamicRange.scalar',
    'CompressionArtifacts.scalar'
]
features_display = [f.replace('.scalar', '') for f in features]  

# Dataset dictionary
datasets = {
    'FFHQ': ffhq_data,
    'LFW': lfw_data,
    'FFHQ_StyleGAN': stylegan_data,
    'LFW_Hyperparameter': lfw_hyperparameter_data
}

# Color mapping
dataset_colors = {
    'FFHQ': 'blue', 
    'LFW': 'red',  
    'FFHQ_StyleGAN': 'green',  
    'LFW_Hyperparameter': 'purple'
}


peak_density_df = pd.DataFrame(index=datasets.keys(), columns=features_display)

# Plot KDE 
for feature, feature_display in zip(features, features_display):
    plt.figure(figsize=(10, 6))

    for label, data in datasets.items():
        if feature in data.columns:
            clean_data = pd.to_numeric(data[feature], errors='coerce').dropna()

            # Dynamic outlier removal (0.5% - 99.5% range)
            min_valid, max_valid = clean_data.quantile([0.005, 0.995])  
            clean_data = clean_data[(clean_data >= min_valid) & (clean_data <= max_valid)]

            if len(clean_data) > 1:
                color = dataset_colors[label] 

               
                kde_plot = sns.kdeplot(
                    clean_data, 
                    fill=True, 
                    bw_adjust=1.5, 
                    alpha=0.6, 
                    label=label, 
                    color=color, 
                    common_norm=True  
                )

               
                if kde_plot.lines and len(kde_plot.lines) > 0:
                    kde_data = kde_plot.lines[0].get_data()
                    x_values, y_values = kde_data[0], kde_data[1]
                    peak_index = np.argmax(y_values)
                    peak_value = x_values[peak_index]

                    
                    peak_density_df.loc[label, feature_display] = round(peak_value, 4)

                    
                    plt.text(peak_value, max(y_values), f'{peak_value:.2f}', 
                             fontsize=10, ha='center', va='bottom', fontweight='bold', color=color)

    
    plt.title(f'OFIQ Feature Distribution for Albedo Images (Without Lighting) - {feature_display}')
    plt.xlabel(feature_display)
    plt.ylabel('Density (Normalized)')
    plt.xlim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.show()


peak_density_df.dropna(how="all", inplace=True)


if not peak_density_df.empty:
    print("\n📊 OFIQ Feature Distribution for Albedo Images (Without Lighting)")
    print(peak_density_df.to_string())
