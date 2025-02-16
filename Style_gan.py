import os
import sys
import numpy as np
import torch
import dnnlib
from PIL import Image


sys.path.append(os.path.join(os.getcwd(), 'stylegan3'))


import legacy

# Path to the pre-trained model
PRETRAINED_MODEL_PATH = 'pretrained_models/stylegan3-r-ffhq-1024x1024.pkl'
OUTPUT_DIR = 'generated_images_all_classes'
os.makedirs(OUTPUT_DIR, exist_ok=True)


color_classes = [
    [246, 237, 228],  # Class 1
    [243, 231, 219],  # Class 2
    [247, 234, 208],  # Class 3
    [234, 218, 186],  # Class 4
    [215, 189, 150],  # Class 5
    [160, 126, 86],   # Class 6
    [130, 92, 67],    # Class 7
    [96, 65, 52],     # Class 8
    [58, 49, 42],     # Class 9
    [41, 36, 32]      # Class 10
]

# Number of images to generate
NUM_IMAGES = 1000
LATENT_DIM = 512

# Set device to GPU or CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained StyleGAN3 model
print("Loading pre-trained StyleGAN3 model...")
with dnnlib.util.open_url(PRETRAINED_MODEL_PATH) as f:
    G = legacy.load_network_pkl(f)['G_ema']  # Generator
    G = G.to(device)  


def recolor_image(image, target_color):
    """Adjust the average color of an image to match a target skin tone."""
    img_np = np.array(image)
    avg_color = np.mean(img_np, axis=(0, 1))  
    color_diff = np.array(target_color) - avg_color
    recolored_img = np.clip(img_np + color_diff, 0, 255).astype(np.uint8)
    return Image.fromarray(recolored_img)


def generate_images_all_classes(generator, num_images, output_dir):
    for i in range(1, num_images + 1):
        
        z = np.random.randn(1, LATENT_DIM).astype(np.float32)
        z_tensor = torch.from_numpy(z).to(device)

      
        with torch.no_grad():
            img = generator(z_tensor, None, noise_mode='const')
        img_np = (img.permute(0, 2, 3, 1).cpu().numpy() * 127.5 + 128).clip(0, 255).astype(np.uint8)[0]
        img_pil = Image.fromarray(img_np, 'RGB')

        # Apply all classes
        for class_idx, target_color in enumerate(color_classes, start=1):
            recolored_img = recolor_image(img_pil, target_color)
            filename = f'{i:05d}_class{class_idx}.png' 
            recolored_img.save(os.path.join(output_dir, filename))
            print(f"Image {i:05d} recolored for Class {class_idx} and saved as {filename}")

# Generate images for all classes
print("Generating 1000 synthetic skin ima/home/ext10812/TRUST/libges with StyleGAN3 for all classes...")
generate_images_all_classes(G, NUM_IMAGES, OUTPUT_DIR)
print("Image generation and multi-class recoloring complete.")
