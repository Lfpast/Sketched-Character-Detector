import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

def visualize_bounding_box(image_path, predicted_bbox=None, groundtruth_bbox=None):
  """Plot the original image, the predicted bounding box, and groundtruth (if any)
  image_path: the path to the image
  predicted_bbox: the predicted bounding box for drawings in the image
  groundtruth_bbox: the ground truth bounding box

  return void
  """
  image = cv2.imread(image_path)
  if image is None:
      print(f"Error: Could not read image at {image_path}")
      return
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  fig, ax = plt.subplots(1, figsize=(5, 5))
  ax.imshow(image)

  def draw_boxes(bbox, color, label):
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x, y-18, f'{label}', fontsize=12, verticalalignment='top', color=color)

  if predicted_bbox is not None:
      draw_boxes(predicted_bbox, 'red', 'Predicted')
  if groundtruth_bbox is not None:
      draw_boxes(groundtruth_bbox, 'green', 'Ground Truth')

  plt.axis('off')
  plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
  
  # Save to utils folder
  utils_dir = os.path.dirname(os.path.abspath(__file__))
  filename = os.path.basename(image_path)
  name, _ = os.path.splitext(filename)
  save_path = os.path.join(utils_dir, f"vis_{name}.png")
  
  plt.savefig(save_path)
  plt.close()
  print(f"Saved visualization to {save_path}")
