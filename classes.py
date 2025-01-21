import json
import os

import cv2 as cv
import numpy as np
import pyautogui
import tensorflow as tf
from numpy.linalg import norm
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (MobileNetV2,
                                                        preprocess_input)
from tensorflow.keras.preprocessing.image import img_to_array


def cosine_distance(a, b):
    """Compute cosine distance between two vectors.
       Returns 1 - cosine_similarity."""
    return 1 - np.dot(a, b) / (norm(a) * norm(b))


class IconMatcher:
    def __init__(self, base_library_dir, input_size=(224, 224), cache_path='library_embeddings.npz'):
        """
        base_library_dir: Path to 'dota2_ability_icons', which has subfolders 
                          [Lowtier, Situational, Tier1, Tier2, Tier3, Tier4].
        input_size: (width, height) to which images will be resized before extracting features.
                    Using (224,224) aligns with MobileNetV2 defaults.
        cache_path: Path to a file where library embeddings will be cached.
        """
        self.base_library_dir = base_library_dir
        self.input_size = input_size
        self.cache_path = cache_path

        # Build the MobileNetV2 feature extractor (no top layers)
        base_model = MobileNetV2(weights='imagenet', include_top=False,
                                 input_shape=(input_size[1], input_size[0], 3))
        self.model = tf.keras.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        self.model.trainable = False

        # Wrap inference in a tf.function for faster repeated calls.
        self.get_embeddings_tf = tf.function(self._get_embeddings_batch)

        # Build or load the library embeddings
        self.library_embeddings = {}
        self.library_keys = []
        if os.path.exists(self.cache_path):
            self._load_library_cache()
        else:
            self._build_library_embeddings()
            self._save_library_cache()
        self._prepare_library_matrix()

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

    def _save_library_cache(self):
        """
        Saves the computed library embeddings and keys to cache_path.
        """
        # Sort the dictionary by key to keep order consistent.
        self.library_keys = sorted(self.library_embeddings.keys())
        # Create a list of embeddings in the same order.
        lib_array = np.array([self.library_embeddings[key]
                             for key in self.library_keys])
        np.savez_compressed(self.cache_path, library_keys=np.array(
            self.library_keys), library_array=lib_array)
        print(f"[INFO] Saved library embeddings cache to {self.cache_path}")

    def _load_library_cache(self):
        """
        Loads library embeddings and keys from cache_path.
        """
        data = np.load(self.cache_path, allow_pickle=True)
        self.library_keys = data['library_keys'].tolist()
        lib_array = data['library_array']
        # Rebuild the dictionary for compatibility (if needed).
        self.library_embeddings = {key: emb for key,
                                   emb in zip(self.library_keys, lib_array)}
        print(f"[INFO] Loaded library embeddings cache from {self.cache_path}")

    def _prepare_library_matrix(self):
        """
        Prepare a normalized library embedding matrix and key list.
        """
        # Build a matrix of embeddings.
        lib_matrix = np.array([self.library_embeddings[key]
                              for key in self.library_keys])
        # Normalize each embedding.
        norms = np.linalg.norm(lib_matrix, axis=1, keepdims=True) + 1e-10
        self.library_matrix = lib_matrix / norms

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
        # Use predict_on_batch so even a single image is treated as a batch.
        embedding = self.model.predict_on_batch(arr)[0]
        return embedding

    def _get_embeddings_batch(self, batch):
        """
        This tf.function-wrapped function accepts a preprocessed batch of images and
        returns their embeddings.
        """
        return self.model(batch)

    def match_icon(self, cropped_pil):
        """
        Given a cropped PIL image, compute its embedding and find the nearest 
        library icon using vectorized cosine distance.
        Returns (best_key, best_distance).
        """
        cropped = cropped_pil.resize(self.input_size, Image.Resampling.LANCZOS)
        arr = img_to_array(cropped)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)
        embedding = self.model.predict_on_batch(arr)[0]

        # Normalize the test embedding.
        embedding_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        # Compute cosine similarities vectorized (library embeddings are already normalized).
        similarities = np.dot(self.library_matrix, embedding_norm)
        distances = 1 - similarities  # cosine distance

        best_idx = np.argmin(distances)
        best_key = self.library_keys[best_idx]
        best_dist = distances[best_idx]
        return best_key, best_dist

    def match_all(self, test_folder, batch_size=32):
        """
        Walk through all images (recursively) in test_folder,
        match each image in batches, and return a dictionary mapping each test image's
        full path to a tuple (matched_library_key, distance).
        """
        # Gather all image paths.
        image_paths = []
        for root, _, files in os.walk(test_folder):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_paths.append(os.path.join(root, file))

        results = {}
        batch_images = []
        batch_paths = []

        for i, path in enumerate(image_paths):
            try:
                img = Image.open(path).convert("RGB")
            except Exception as e:
                print(f"[WARNING] Skipping {path}: {e}")
                continue
            # Resize and convert the image.
            img_resized = img.resize(self.input_size, Image.Resampling.LANCZOS)
            arr = img_to_array(img_resized)
            batch_images.append(arr)
            batch_paths.append(path)

            # Process batch if full or if it's the last image.
            if len(batch_images) == batch_size or i == len(image_paths) - 1:
                batch_array = np.array(batch_images)
                batch_array = preprocess_input(batch_array)
                # Use tf.function-wrapped batch inference.
                batch_embeddings = self.get_embeddings_tf(batch_array)
                batch_embeddings = batch_embeddings.numpy()  # Convert to numpy array
                # Normalize embeddings.
                norms = np.linalg.norm(
                    batch_embeddings, axis=1, keepdims=True) + 1e-10
                batch_embeddings_norm = batch_embeddings / norms

                # Vectorized search for nearest library icon.
                for idx, embedding_norm in enumerate(batch_embeddings_norm):
                    similarities = np.dot(self.library_matrix, embedding_norm)
                    distances = 1 - similarities
                    best_idx = np.argmin(distances)
                    best_key = self.library_keys[best_idx]
                    best_dist = distances[best_idx]
                    results[batch_paths[idx]] = (best_key, best_dist)
                    print(
                        f"[INFO] {batch_paths[idx]} matched with {best_key} (distance: {best_dist:.4f})")

                batch_images, batch_paths = [], []
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
