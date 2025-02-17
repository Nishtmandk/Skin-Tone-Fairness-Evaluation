
# Skin Tone Fairness Evaluation

## DTU Master Thesis

This project is part of the research "Skin Tone Fairness Evaluation of Face Image Quality Assessment Algorithms based on 3D Morphable Face Models". The objective is to analyze biases in skin tone classification within biometric systems. The study involves different color-based classification methods using predefined RGB color categories from the Google SkinTone Guide.

![Monk Skin Tone Pipeline](https://github.com/Nishtmandk/Skin-Tone-Fairness-Evaluation/blob/main/Monk.jpg?raw=true)

## Required Steps Before Running This Project

Before running this project, you must first install and set up the TRUST model.

## Step 1: Clone and Install the TRUST Model

TRUST (Toward a Racially Unbiased Skin Tone Estimation) is required for this project. First, clone and set up TRUST:

git clone https://github.com/HavenFeng/TRUST
cd TRUST
pip install -r requirements.txt

Next, run TRUST to estimate albedo and reconstruct facial geometry:

python process_images.py --dataset ffhq

## Step 2: Clone This Repository and Run the Evaluation Code

Once TRUST is installed and processed the dataset, clone this repository and run the skin tone fairness evaluation:

cd ..
git clone https://github.com/Nishtmandk/Skin-Tone-Fairness-Evaluation.git
cd Skin-Tone-Fairness-Evaluation

Now, install the necessary dependencies:

pip install -r requirements.txt

## Step 3: Run the Bias Evaluation Scripts

After setting up TRUST and cloning this repository, execute the bias evaluation:

python analyze_bias.py

This script compares different skin tone classification models and calculates bias scores.

## Methodology

This project utilizes TRUST(https://github.com/HavenFeng/TRUST) to reconstruct facial geometry using FLAME 3D models(https://github.com/HavenFeng/photometric_optimization?tab=readme-ov-file) and estimates albedo to separate intrinsic skin properties from lighting effects.

## Classification Models Used

Weighted Grayscale - Enhances accuracy by adjusting grayscale representation based on human visual perception.

HSV Color Space - Segments images based on hue, saturation, and value.

YCbCr Color Space - Separates luminance from chrominance for better skin tone differentiation.

Average Grayscale - Computes the mean of RGB values for skin tone estimation.

## Key Findings

Bias scores varied across datasets and classification methods.

Highest bias was observed in LFW Hyperparameter - Weighted Grayscale (1.658).

Lowest bias was found in FFHQ StyleGAN - YCbCr (0.801).

Weighted Grayscale exhibited more bias than YCbCr and HSV models.

The ideal distribution assumption (10 percent per class) was not met in any dataset.

## Folder Structure

### Required Folders Before Running TRUST

Ensure the following folders exist before running TRUST:

ffhq-val/ - Sorted images

ffhq-val-amks/ - Generated landmarks

ffhq-val_val_list.csv - List of image landmarks

ffhq_outputs/ - Processing results

### TRUST Model Outputs

Once TRUST is executed, processed outputs will be stored in:

inputs/ - Raw input images

predicted_images/ - Images with lighting effects

albedo/ - Extracted textures from images

albedo_images/ - Images without lighting effects

## Results and Visualizations

Python_GPU_code        
https://drive.google.com/file/d/1naJDFeItUc2g2_jmi13zfVpiBAu_KxF8/view?usp=sharing

Python_code            
https://drive.google.com/file/d/1qnqyjYsNQuG3dwxeh0wG-zfFuxd_Ab8l/view?usp=sharing

lfw_Hyperparameter     
https://drive.google.com/file/d/1hkGCG92nRXHcsq0t8P144xLL_nI4WfJP/view?usp=sharing

ffhq_Style_GAN         
https://drive.google.com/file/d/1DGLOk8oe3N8Iq3bBDfELYaHJFuPhKm-i/view?usp=sharing

lfw                    
https://drive.google.com/file/d/1kcVQUueHI-_JJfASFrI75YZIi0qz3ul7/view?usp=sharing

ffhq                  
https://drive.google.com/file/d/1Euvte3UkCex9zF_W37mgOIhbsR8cOubl/view?usp=sharing

Images



### Skin Tone Fairness Evaluation

![Skin Tone Fairness](Work.png)https://drive.google.com/file/d/1I2tMDZc9VQI0vj0afywuVksJ0WsPOzIA/view?usp=sharing

### TRUST Model Output Example

![TRUST Model](MST.png)https://drive.google.com/file/d/1mJJqgCzGYx6a0igmWjZKELrPTAUBfxIg/view?usp=sharing

## Author

Nishtman Andisha

## License

This project is published under the DTU Computer Science and Engineering Department, 2025.
EOF
