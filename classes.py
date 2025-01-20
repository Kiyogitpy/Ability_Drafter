import json
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pyautogui
import tensorflow as tf
from numpy.linalg import norm
from PIL import Image
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array


def cosine_distance(a, b):
    """Compute cosine distance between two vectors.
       Returns 1 - cosine_similarity."""
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


class IconMatcher:
    def __init__(self, base_library_dir, input_size=(224, 224)):
        """
        base_library_dir: Path to 'dota2_ability_icons', which has subfolders 
                          [Lowtier, Situational, Tier1, Tier2, Tier3, Tier4].
        input_size: (width, height) to which images will be resized before extracting features.
                    Using (224,224) aligns with MobileNetV2 defaults.
        """
        self.base_library_dir = base_library_dir
        self.input_size = input_size

        # Build a MobileNetV2 feature extractor (no top layers)
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=(input_size[0], input_size[1], 3))
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        self.model.trainable = False

        self.library_embeddings = {}
        self._build_library_embeddings()

    def _build_library_embeddings(self):
        """
        Walk through each subfolder in the library directory and compute
        embeddings for each image. Keys are in the form "TierName/filename.png".
        """
        for tier_name in os.listdir(self.base_library_dir):
            tier_path = os.path.join(self.base_library_dir, tier_name)
            if not os.path.isdir(tier_path):
                continue
            for fname in os.listdir(tier_path):
                if not fname.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue
                fpath = os.path.join(tier_path, fname)
                embedding = self._get_embedding(fpath)
                relative_key = os.path.join(tier_name, fname)
                self.library_embeddings[relative_key] = embedding
        print(
            f"[INFO] Built embeddings for {len(self.library_embeddings)} library icons.")

    def _get_embedding(self, img_path):
        """
        Loads an image, resizes it to self.input_size using LANCZOS resampling,
        and returns its embedding from the MobileNetV2 feature extractor.
        """
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.input_size, Image.Resampling.LANCZOS)
        arr = img_to_array(img)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        embedding = self.model.predict(arr)[0]
        return embedding

    def match_icon(self, cropped_pil):
        """
        Given a cropped PIL image, compute its embedding and find the nearest 
        library icon using cosine distance.
        Returns (best_key, best_distance).
        """
        cropped = cropped_pil.resize(self.input_size, Image.Resampling.LANCZOS)
        arr = img_to_array(cropped)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        embedding = self.model.predict(arr)[0]

        best_key = None
        best_dist = float('inf')
        for lib_key, lib_emb in self.library_embeddings.items():
            dist = cosine_distance(embedding, lib_emb)  # using cosine distance
            if dist < best_dist:
                best_dist = dist
                best_key = lib_key
        return best_key, best_dist

    def match_all(self, test_folder):
        """
        Walk through all images (recursively) in test_folder,
        match each image, and return a dictionary mapping each test image's
        full path to a tuple (matched_library_key, distance).
        """
        results = {}
        for root, _, files in os.walk(test_folder):
            for file in files:
                if not file.lower().endswith(('png', 'jpg', 'jpeg')):
                    continue
                full_path = os.path.join(root, file)
                try:
                    img = Image.open(full_path).convert("RGB")
                except Exception as e:
                    print(f"[WARNING] Skipping {full_path}: {e}")
                    continue
                best_key, dist = self.match_icon(img)
                results[full_path] = (best_key, dist)
                print(
                    f"[INFO] {full_path} matched with {best_key} (distance: {dist:.4f})")
        return results


def create_pair_image(test_path, lib_path, desired_size):
    """
    Creates a side-by-side image from the test image and library image.
    Both images are resized to desired_size.
    Returns a PIL Image.
    """
    test_img = Image.open(test_path).convert("RGB").resize(
        desired_size, Image.Resampling.LANCZOS)
    lib_img = Image.open(lib_path).convert("RGB").resize(
        desired_size, Image.Resampling.LANCZOS)
    # Create a new image with combined width.
    pair_width = desired_size[0] * 2
    pair_height = desired_size[1]
    pair_img = Image.new("RGB", (pair_width, pair_height))
    pair_img.paste(test_img, (0, 0))
    pair_img.paste(lib_img, (desired_size[0], 0))
    return pair_img


def build_mosaic(pair_images, pairs_per_row, pair_size):
    """
    Arranges a list of pair_images into a mosaic grid.
    Each pair_image is of size pair_size.
    Returns a PIL Image of the mosaic.
    """
    num_pairs = len(pair_images)
    num_rows = (num_pairs + pairs_per_row - 1) // pairs_per_row
    mosaic_width = pairs_per_row * pair_size[0]
    mosaic_height = num_rows * pair_size[1]
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    for i, pair_img in enumerate(pair_images):
        row = i // pairs_per_row
        col = i % pairs_per_row
        x = col * pair_size[0]
        y = row * pair_size[1]
        mosaic.paste(pair_img, (x, y))
    return mosaic


class ImageCropper:
    def __init__(self, json_file, screenshot=True, image_path=None):
        # Load the JSON file containing object coordinates
        with open(json_file, "r") as file:
            self.objects = json.load(file)

        # Capture a screenshot using pyautogui or load an image if needed
        if screenshot:
            img = pyautogui.screenshot()
            img = np.array(img)
            self.img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        else:
            # Load image from file
            if image_path:
                self.img = cv.imread(image_path)
            else:
                self.img = None  # No image to load

        # Dictionary to store cropped images
        self.image_dict = {}

    def crop_images(self):
        # Iterate through the objects and crop regions
        for obj_name, obj in self.objects.items():
            x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]

            # Crop the region and save it in the image_dict
            cropped_img = self.img[y:y + h, x:x + w]
            self.image_dict[obj_name] = cropped_img

    def get_image_dict(self):
        return self.image_dict
