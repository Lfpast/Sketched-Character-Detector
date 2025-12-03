import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from random import randint
from utils.visualization import visualize_bounding_box

def extract_failure_case(loss_func, pred, gt, k=5):
  """ Extract failure cases with the worst performance.
  model: the trained model
  loss_func: a loss function similar to our previous implementation
  pred: the predicted bounding box (scaled to the image size)
  gt: the ground truth bbox in the test dataset
  k: the number of top failures with the largest loss value

  return the index to the top 5 failure case
  """
  loss = np.float32([loss_func(pred[i],gt[i]) for i in range(len(pred))])
  failure_case_indexes = np.argsort(loss)[-k:]
  return failure_case_indexes

def rescale_bbox(src, raw_pred, scaled_gt):
  original_image = load_img(src, target_size=None)
  original_width = original_image.width
  original_height = original_image.height
  def _rescale_bbox(bbox):
    x, y, w, h = bbox
    x = x * original_width
    y = y * original_height
    w = w * original_width
    h = h * original_height
    return np.array([x, y, w, h])
  return _rescale_bbox(raw_pred), _rescale_bbox(scaled_gt)

def predict_bounding_box(model, src):
  """leverage the model to identify drawings in an image
  model: a trained object detector model
  src: the source of the image file

  return [x, y, w , h] for the predicted bounding box in the original image
  """
  original_image = load_img(src, target_size=None)
  original_width = original_image.width
  original_height = original_image.height
  image = original_image.resize((224, 224))
  image = img_to_array(image)

  image = np.expand_dims(image, axis=0)  # Add a dimension to fit model input requirements
  image = preprocess_input(image)  # Preprocess image

  # Use model for prediction
  pred = model.predict(image)

  # Scale predicted bounding box coordinates back to original image size
  x_pred, y_pred, w_pred, h_pred = pred[0]
  x_pred *= original_width
  y_pred *= original_height
  w_pred *= original_width
  h_pred *= original_height

  return [x_pred, y_pred, w_pred, h_pred]

def inspect(model, dataset, k=None):
  """Visually inspect one instance with its predicted and ground truth bbox
  model: the trained model
  dataset: the full dataset with image source and ground truth
  k: the index of the image under inspection

  return void
  """
  if k is None:
    k = randint(0, len(dataset)-1)
  
  item_src = dataset[k]['src']
  
  # Path resolution logic
  data_dir = './data'
  # Try default
  src = os.path.join(data_dir, item_src)
  
  if not os.path.exists(src):
      filename = os.path.basename(item_src)
      # Try training_images
      alt_path = os.path.join(data_dir, 'training_images', filename)
      if os.path.exists(alt_path):
          src = alt_path
      else:
          # Try testing_images
          alt_path = os.path.join(data_dir, 'testing_images', filename)
          if os.path.exists(alt_path):
              src = alt_path

  ground_truth = dataset[k]['bbox']
  pred = predict_bounding_box(model, src)
  visualize_bounding_box(src, pred, ground_truth)

def generate_grading_files(model, grading_paths_file, testing_paths_file):
    data_dir = './data'
    
    with open(grading_paths_file, 'rb') as f:
        paths = pickle.load(f)
    # Resolve paths: prepend data_dir if file doesn't exist at relative path
    resolved_paths = [os.path.join(data_dir, p) if not os.path.exists(p) else p for p in paths]
    grading_pred = np.array([predict_bounding_box(model, path) for path in resolved_paths])
    
    with open('grading_pred.pkl', 'wb') as f:
        pickle.dump(grading_pred, f, pickle.HIGHEST_PROTOCOL)

    with open(testing_paths_file, 'rb') as f:
        paths = pickle.load(f)
    # Resolve paths
    resolved_paths = [os.path.join(data_dir, p) if not os.path.exists(p) else p for p in paths]
    testing_pred = np.array([predict_bounding_box(model, path) for path in resolved_paths])
    
    with open('testing_pred.pkl', 'wb') as f:
        pickle.dump(testing_pred, f, pickle.HIGHEST_PROTOCOL)
    
    print("Generated grading_pred.pkl and testing_pred.pkl")
