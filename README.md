cat << EOF > README.md
# Skin Tone Fairness Evaluation

## DTU Master Thesis

This project is part of the research "Skin Tone Fairness Evaluation of Face Image Quality Assessment Algorithms based on 3D Morphable Face Models". The objective is to analyze biases in skin tone classification within biometric systems. The study involves different color-based classification methods using predefined RGB color categories from the Google SkinTone Guide.

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

This project utilizes TRUST to reconstruct facial geometry using FLAME 3D models and estimates albedo to separate intrinsic skin properties from lighting effects.

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

### Skin Tone Fairness Evaluation

![Skin Tone Fairness](Work.png)

### TRUST Model Output Example

![TRUST Model](MST.png)

## Author

Nishtman Andisha

## License

This project is published under the DTU Computer Science and Engineering Department, 2025.
EOF
