import os
import cv2
import numpy as np
import argparse
import csv
from tqdm import tqdm
from scipy.spatial.distance import euclidean

# Monk skin tone RGB values
monk_rgb_tones = {
    1: (239, 208, 207), 2: (234, 192, 165), 3: (221, 178, 136), 4: (200, 150, 108),
    5: (184, 124, 93), 6: (161, 96, 67), 7: (142, 74, 50), 8: (120, 54, 38),
    9: (94, 38, 24), 10: (69, 28, 16)
}

def calculate_rgb_hsv_ycbcr_metrics(image):
    """ Extract RGB, Grayscale, HSV, and YCbCr color metrics while ignoring black pixels """
    
    # Create a mask to ignore black pixels
    mask = np.any(image > 0, axis=-1)

    # Check if the image is completely black
    if np.count_nonzero(mask) == 0:
        return None, None, None, None, None, None, None, None, None, None, None

    # Extract only non-black pixels
    valid_pixels = image[mask]

    avg_r = valid_pixels[:, 2].mean()
    avg_g = valid_pixels[:, 1].mean()
    avg_b = valid_pixels[:, 0].mean()

    avg_gray = valid_pixels.mean(axis=1).mean()
    weighted_gray = (0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_valid_pixels = hsv_image[mask]
    avg_h = hsv_valid_pixels[:, 0].mean()
    avg_s = hsv_valid_pixels[:, 1].mean()
    avg_v = hsv_valid_pixels[:, 2].mean()

    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycbcr_valid_pixels = ycbcr_image[mask]
    avg_y = ycbcr_valid_pixels[:, 0].mean()
    avg_cb = ycbcr_valid_pixels[:, 1].mean()
    avg_cr = ycbcr_valid_pixels[:, 2].mean()

    return avg_r, avg_g, avg_b, avg_gray, weighted_gray, avg_h, avg_s, avg_v, avg_y, avg_cb, avg_cr

def process_images(input_folder, output_folder):
    """ Process all images in the 'skin' folder and extract color space features """
    os.makedirs(output_folder, exist_ok=True)
    csv_path = os.path.join(output_folder, "color_metrics.csv")

    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "Image Name", "Average R", "Average G", "Average B", "Average Grayscale", "Weighted Grayscale",
            "Hue (H)", "Saturation (S)", "Value (V)", "Luma (Y)", "Chroma-Blue (Cb)", "Chroma-Red (Cr)",
            "Closest Monk Tone RGB", "Closest Monk Tone Average Grayscale", "Closest Monk Tone Weighted Grayscale",
            "Closest Monk Tone HSV", "Closest Monk Tone YCbCr"
        ])

        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        with tqdm(total=len(image_files), desc="Processing Images", unit="image") as pbar:
            for filename in image_files:
                input_path = os.path.join(input_folder, filename)
                image = cv2.imread(input_path)
                if image is None:
                    print(f"Could not load image: {input_path}")
                    continue
                
                # Compute color metrics (ignoring black background)
                metrics = calculate_rgb_hsv_ycbcr_metrics(image)
                
                if metrics[0] is None:
                    print(f"Skipping {filename} due to all-black pixels.")
                    continue

                avg_r, avg_g, avg_b, avg_gray, weighted_gray, avg_h, avg_s, avg_v, avg_y, avg_cb, avg_cr = metrics
                
                # Find closest Monk skin tone
                closest_rgb_tone = min(monk_rgb_tones, key=lambda t: euclidean((avg_r, avg_g, avg_b), monk_rgb_tones[t]))
                closest_gray_tone = min(monk_rgb_tones, key=lambda t: abs(avg_gray - sum(monk_rgb_tones[t]) / 3))
                closest_weighted_gray_tone = min(monk_rgb_tones, key=lambda t: abs(weighted_gray - (0.299 * monk_rgb_tones[t][0] + 0.587 * monk_rgb_tones[t][1] + 0.114 * monk_rgb_tones[t][2])))
                
                monk_hsv = {t: cv2.cvtColor(np.uint8([[monk_rgb_tones[t]]]), cv2.COLOR_RGB2HSV)[0][0] for t in monk_rgb_tones}
                closest_hsv_tone = min(monk_hsv, key=lambda t: euclidean((avg_h, avg_s, avg_v), monk_hsv[t]))

                monk_ycbcr = {t: cv2.cvtColor(np.uint8([[monk_rgb_tones[t]]]), cv2.COLOR_RGB2YCrCb)[0][0] for t in monk_rgb_tones}
                closest_ycbcr_tone = min(monk_ycbcr, key=lambda t: euclidean((avg_y, avg_cb, avg_cr), monk_ycbcr[t]))

                # Save to CSV
                csv_writer.writerow([
                    filename, f"{avg_r:.2f}", f"{avg_g:.2f}", f"{avg_b:.2f}", f"{avg_gray:.2f}", f"{weighted_gray:.2f}",
                    f"{avg_h:.2f}", f"{avg_s:.2f}", f"{avg_v:.2f}", f"{avg_y:.2f}", f"{avg_cb:.2f}", f"{avg_cr:.2f}",
                    closest_rgb_tone, closest_gray_tone, closest_weighted_gray_tone, closest_hsv_tone, closest_ycbcr_tone
                ])
                pbar.update(1)

    print(f"Processing complete! CSV saved at: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract color metrics from skin images")
    parser.add_argument("--input_folder", type=str, required=True, help="/home/ext10812/TRUST/ffhq_outputs/0skin/skins/")
    parser.add_argument("--output_folder", type=str, required=True, help="/home/ext10812/TRUST/ffhq_outputs/0skin")
    
    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder)
    print(" All images processed successfully!")