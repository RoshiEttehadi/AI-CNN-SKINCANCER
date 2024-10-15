import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from pathlib import Path

# Set image size
IMG_SIZE = 224

# Get current project path
# ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
benign_folder = os.path.join(Path(ROOT_DIR).parent, 'datasets', 'ISIC-images-benign')
malignant_folder = os.path.join(Path(ROOT_DIR).parent, 'datasets', 'ISIC-images-malignant')

# Load images and label them
def load_images(image_dir, label):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            # Load the image and resize it
            image = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
            image = img_to_array(image) / 255.0  # Normalize image to [0, 1]
            images.append(image)
            labels.append(label)
    return images, labels

# Load malignant and benign images
def load_data():

    benign_images, benign_labels = load_images(benign_folder, 0)
    malignant_images, malignant_labels = load_images(malignant_folder, 1)

    # Combine the benign and malignant images and labels
    images = benign_images + malignant_images
    labels = benign_labels + malignant_labels

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    # Split the dataset into training and validation sets (80-20 split)
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
