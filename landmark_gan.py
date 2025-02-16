import cv2
import os
import numpy as np

# Paths
image_folder = "/home/ext10812/TRUST/FFHQ/ffhq-val"   # Input images folder
output_folder = "/home/ext10812/TRUST/FFHQ/fFhq-val-lmks"  # Output landmarks folder
no_face_log = "no_face_detected_opencv.txt"  # Log file for images without faces

# Haar Cascade Classifier path (pretrained OpenCV model)
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


if not os.path.exists(haar_cascade_path):
    print(f"Haar cascade not found at {haar_cascade_path}")
    exit()

# Load Haar Cascade model
face_cascade = cv2.CascadeClassifier(haar_cascade_path)


os.makedirs(output_folder, exist_ok=True)

# Process each image
for image_name in os.listdir(image_folder):
    if image_name.endswith(".png") or image_name.endswith(".jpg"):  
        image_path = os.path.join(image_folder, image_name)
        output_path = os.path.join(output_folder, image_name.rsplit(".", 1)[0] + ".npy")

       
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
          
            landmarks_array = []
            for (x, y, w, h) in faces:
                landmarks_array.append([x, y])  # Top-left corner
                landmarks_array.append([x + w, y])  # Top-right corner
                landmarks_array.append([x, y + h])  # Bottom-left corner
                landmarks_array.append([x + w, y + h])  # Bottom-right corner
                break  # 

            # Save landmarks to file
            np.save(output_path, np.array(landmarks_array))
            print(f"Landmarks saved: {output_path}")
        else:
           
            print(f"No face detected: {image_path}")
            with open(no_face_log, "a") as log_file:
                log_file.write(f"{image_name}\n")
import cv2
import os
import numpy as np

# Paths
image_folder = "/home/ext10812/TRUST/FFHQ/ffhq-val"   # Input images folder
output_folder = "/home/ext10812/TRUST/FFHQ/ffhq-val-lmks"  # Output landmarks folder
no_face_log = "no_face_detected_opencv.txt"  # Log file for images without faces

# Haar Cascade Classifier path (pretrained OpenCV model)
haar_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


face_cascade = cv2.CascadeClassifier(haar_cascade_path)


os.makedirs(output_folder, exist_ok=True)


for image_name in os.listdir(image_folder):
    if image_name.endswith(".png"):
        image_path = os.path.join(image_folder, image_name)
        output_path = os.path.join(output_folder, image_name.replace(".png", ".npy"))

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading: {image_path}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
          
            landmarks_array = []
            for (x, y, w, h) in faces:
                landmarks_array.append([x, y])  # Top-left corner
                landmarks_array.append([x + w, y])  # Top-right corner
                landmarks_array.append([x, y + h])  # Bottom-left corner
                landmarks_array.append([x + w, y + h])  # Bottom-right corner
                break 

            
            np.save(output_path, np.array(landmarks_array))
            print(f"Landmarks saved: {output_path}")
        else:
            
            print(f"No face detected: {image_path}")
            with open(no_face_log, "a") as log_file:
                log_file.write(f"{image_name}\n")
