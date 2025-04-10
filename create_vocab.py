import os
from pathlib import Path
import cv2
import numpy as np
from config import SETTINGS


VOCAB_SIZE = SETTINGS["place_recognition"]["vocab_size"]


def create_vocabulary():
    # Check if the vocabulary already exists and don't overwrite
    save_path = Path("vocabulary/kitti_cv2.npy")
    if os.path.exists(save_path):
        print(f"Vocabulary {save_path} already exists!")
        return

    # Initialize the ORB feature detector
    orb = cv2.ORB_create()

    # Create a BOW trainer and set the number of clusters to 1000 visual words
    bow_trainer = cv2.BOWKMeansTrainer(VOCAB_SIZE)

    # Define the base directories and image subdirectories
    base_paths = [
        Path("/home/panos/Documents/data/kitti/00/"),
        Path("/home/panos/Documents/data/kitti/06/")
    ]
    sub_dirs = ["image_0", "image_1"]

    # Gather all image paths from the specified directories (assuming images are PNG files)
    image_paths = []
    for base in base_paths:
        for sub in sub_dirs:
            # Construct the subdirectory path
            dir_path = base / sub
            # Append all .png files from this subdirectory to the list
            image_paths.extend(list(dir_path.glob("*.png")))

    print(f"Found {len(image_paths)} images for vocabulary creation.")

    # Loop over each image to accumulate descriptors
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error reading:", path)
            continue

        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(image, None)
        if descriptors is not None:
            descriptors = np.float32(descriptors)
            bow_trainer.add(descriptors)

    # Cluster all accumulated descriptors to form the vocabulary
    vocabulary = bow_trainer.cluster()

    # Save the vocabulary for future use
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, vocabulary)
    print("Vocabulary shape:", vocabulary.shape)

if __name__ == "__main__":
    create_vocabulary()
