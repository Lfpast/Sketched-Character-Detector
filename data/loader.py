import numpy as np
import json
import os
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def download_dataset(data_dir):
    """Download and extract the dataset."""
    # Dataset is already provided in the data folder
    pass

def load_json_dataset(filepath, limit=None):
    """Load the dataset from a JSON file."""
    with open(filepath) as f:
        dataset = json.load(f)
    if limit:
        dataset = dataset[:limit]
    return dataset

def preprocess_dataset(dataset, data_dir, image_size=(224, 224)):
  """Preprocess the raw dataset
  dataset: the parsed JSON file as the dataset
  data_dir: directory where images are stored
  image_size: targeted image size for the model input

  return an array of the image paths, the transformed images and ground truth bboxes
  """
  images = []
  bboxes = []
  urls = []

  for item in dataset:
    # item['src'] is like "images/1.png"
    # We need to join it with data_dir
    image_path = os.path.join(data_dir, item['src'])
    
    # Check if image exists, if not try other locations (training_images/testing_images)
    if not os.path.exists(image_path):
        filename = os.path.basename(item['src'])
        # Try training_images
        alt_path = os.path.join(data_dir, 'training_images', filename)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            # Try testing_images
            alt_path = os.path.join(data_dir, 'testing_images', filename)
            if os.path.exists(alt_path):
                image_path = alt_path

    urls.append(image_path)
    
    original_image = load_img(image_path, target_size=None)
    original_width = original_image.width
    original_height = original_image.height
    image = original_image.resize(image_size)
    image = img_to_array(image)
    image = preprocess_input(image)
    images.append(image)

    bbox = item['bbox']
    x, y, width, height = bbox

    # Normalize the bbox into range [0, 1] according to the proportion to (width, height)
    x_new = x / original_width
    y_new = y / original_height
    new_width = width / original_width
    new_height = height / original_height
    list = [x_new,y_new,new_width,new_height]
    bboxes.append(list)

  urls = np.array(urls, dtype=str)
  images = np.array(images, dtype=np.float32)
  bboxes = np.array(bboxes, dtype=np.float32)
  return urls, images, bboxes

def split_dataset(urls, x, gt, test_split_ratio=0.2):
    """Split the dataset according to the test split ratio
    urls: the input image paths
    x: the input image data (np.ndarray) to be fed into model
    gt: the ground truth boundinng box (np.ndarray)
    test_split_ratio: the percentage of test dataset size in respect to the full dataset

    return the train_url, train_x, train_y, and test_url, test_x, and test_y
    """
    n = x.shape[0]
    test_size = int( n * test_split_ratio)

    # Split the data into train data and test data according to test_size
    test_url, train_url = urls[:test_size], urls[test_size:]
    test_x, train_x = x[:test_size], x[test_size:]
    test_y, train_y = gt[:test_size], gt[test_size:]

    return train_url, train_x, train_y, test_url, test_x, test_y
