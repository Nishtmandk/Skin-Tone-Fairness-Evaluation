
import os
import cv2
import dlib
import numpy as np
from tqdm import tqdm
import argparse
import csv
from scipy.spatial.distance import euclidean

# Dlib face detector and landmark predictor
model_path = "/home/ext10812/TRUST/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(model_path)

# Monk skin tone RGB values
monk_rgb_tones = {
    1: (239, 208, 207), 2: (234, 192, 165), 3: (221, 178, 136), 4: (200, 150, 108),
    5: (184, 124, 93), 6: (161, 96, 67), 7: (142, 74, 50), 8: (120, 54, 38),
    9: (94, 38, 24), 10: (69, 28, 16)
}

def extract_face_skin(image, landmarks, mask_output_path):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # Define face contour
    face_contour = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(17)], dtype=np.int32)
    cv2.fillPoly(mask, [face_contour], 255)

    # Exclude eyes and mouth
    exclude_regions = {
        "eyes": list(range(36, 48)),
        "mouth_outer": list(range(48, 60)),
        "mouth_inner": list(range(60, 68))
    }
    for region, points in exclude_regions.items():
        for i in points:
            cv2.circle(mask, (landmarks.part(i).x, landmarks.part(i).y), 8, 0, -1)

    # Save mask
    cv2.imwrite(mask_output_path, mask)
    return cv2.bitwise_and(image, image, mask=mask)

def calculate_rgb_hsv_ycbcr_metrics(image, mask):
    masked_pixels = image[mask > 0]
    avg_r = masked_pixels[:, 2].mean()
    avg_g = masked_pixels[:, 1].mean()
    avg_b = masked_pixels[:, 0].mean()
    avg_gray = masked_pixels.mean(axis=1).mean()
    weighted_gray = (0.299 * masked_pixels[:, 2] + 0.587 * masked_pixels[:, 1] + 0.114 * masked_pixels[:, 0]).mean()
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    masked_hsv = hsv_image[mask > 0]
    avg_h = masked_hsv[:, 0].mean()
    avg_s = masked_hsv[:, 1].mean()
    avg_v = masked_hsv[:, 2].mean()
    
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    masked_ycbcr = ycbcr_image[mask > 0]
    avg_y = masked_ycbcr[:, 0].mean()
    avg_cb = masked_ycbcr[:, 1].mean()
    avg_cr = masked_ycbcr[:, 2].mean()
    
    return avg_r, avg_g, avg_b, avg_gray, weighted_gray, avg_h, avg_s, avg_v, avg_y, avg_cb, avg_cr

def process_images(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    skin_folder = os.path.join(output_folder, "skins")
    mask_folder = os.path.join(output_folder, "masks")
    os.makedirs(skin_folder, exist_ok=True)
    os.makedirs(mask_folder, exist_ok=True)

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
                output_skin_path = os.path.join(skin_folder, f"skin_{filename}")
                output_mask_path = os.path.join(mask_folder, f"mask_{filename}")

                image = cv2.imread(input_path)
                if image is None:
                    print(f"Could not load image: {input_path}")
                    continue

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)
                for face in faces:
                    landmarks = predictor(gray, face)
                    face_skin = extract_face_skin(image, landmarks, output_mask_path)
                    cv2.imwrite(output_skin_path, face_skin)
                    mask = cv2.imread(output_mask_path, cv2.IMREAD_GRAYSCALE)
                    avg_r, avg_g, avg_b, avg_gray, weighted_gray, avg_h, avg_s, avg_v, avg_y, avg_cb, avg_cr = calculate_rgb_hsv_ycbcr_metrics(face_skin, mask)
                    closest_tone = min(monk_rgb_tones, key=lambda t: euclidean((avg_r, avg_g, avg_b), monk_rgb_tones[t]))
                    csv_writer.writerow([filename, f"{avg_r:.2f}", f"{avg_g:.2f}", f"{avg_b:.2f}", f"{avg_gray:.2f}", f"{weighted_gray:.2f}", f"{avg_h:.2f}", f"{avg_s:.2f}", f"{avg_v:.2f}", f"{avg_y:.2f}", f"{avg_cb:.2f}", f"{avg_cr:.2f}", closest_tone])
                pbar.update(1)
    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract face skin and analyze colors using Dlib")
    parser.add_argument("--input_folder", type=str, default="/home/ext10812/TRUST/ffhq_outputs/albedo_images",
                        help="Path to input folder (default: /home/ext10812/TRUST/ffhq_outputs/albedo_images)")
    parser.add_argument("--output_folder", type=str, default="/home/ext10812/TRUST/ffhq_outputs/0skin",
                        help="Path to output folder (default: /home/ext10812/TRUST/ffhq_outputs/0skin)")
    args = parser.parse_args()

    process_images(args.input_folder, args.output_folder)
    print("All images processed successfully with Dlib.")
