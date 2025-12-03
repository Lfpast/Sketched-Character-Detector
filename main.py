import os
# Suppress warnings and logs
import warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# Disable XLA to avoid CUDNN errors
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
warnings.filterwarnings('ignore')

import tensorflow as tf

# Configure GPU memory growth to avoid CUDNN_STATUS_INTERNAL_ERROR
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

from data.loader import load_json_dataset, preprocess_dataset, split_dataset, download_dataset
from model.network import create_model
from model.losses import loss_func
from train.trainer import train_model, export_proof
from evaluate.evaluator import extract_failure_case, rescale_bbox, inspect, generate_grading_files
from utils.visualization import visualize_bounding_box

def main():
    # Check whether a GPU environment is enabled
    device_name = tf.test.gpu_device_name()
    if len(device_name) > 0:
        print("Found GPU at: {}".format(device_name))
    else:
        print("No GPU detected by TensorFlow. If you have a GPU and want to use it, please ensure NVIDIA drivers and CUDA/CuDNN are installed correctly. Run './gpu_check.sh' for diagnostics.")

    # Configuration
    N = 2000  # we only load N images for the assignment (max=2,000)
    IMAGE_SIZE = (224, 224) # ResNet standard image size
    DATA_DIR = './data'
    DATASET_FILE = os.path.join(DATA_DIR, 'gt_bbox.json')
    SPLIT_TEST_RATIO = 0.2
    N_EPOCH = 50 # You may reduce this for testing

    # 1. Data Loading
    print("Downloading and loading data...")
    download_dataset(DATA_DIR)
    
    if not os.path.exists(DATASET_FILE):
        print(f"Error: Dataset file {DATASET_FILE} not found.")
        return

    dataset = load_json_dataset(DATASET_FILE, limit=N)
    urls, images, bboxes = preprocess_dataset(dataset, DATA_DIR, image_size=IMAGE_SIZE)
    train_url, train_x, train_y, test_url, test_x, test_y = split_dataset(urls, images, bboxes, SPLIT_TEST_RATIO)
    print(f"Data split: {len(train_x)} training samples, {len(test_x)} testing samples.")

    # 2. Model Compilation
    print("Creating and compiling model...")
    model = create_model()
    model.compile(optimizer='rmsprop', loss=loss_func)

    # 3. Training
    print("Starting training...")
    train_model(model, train_x, train_y, epochs=N_EPOCH, batch_size=8)
    export_proof(model)

    # 4. Evaluation
    print("Evaluating model...")
    results = model.evaluate(test_x, test_y, batch_size=8)
    print(f'The loss in the model: {results}')

    print("Extracting failure cases...")
    test_pred = model.predict(test_x, batch_size=8)
    n_failures = 10 # Reduced for demo
    worst_case_indexes = extract_failure_case(loss_func, test_pred, test_y, n_failures)
    
    # Visualize failure cases (optional, might need GUI environment)
    for k in range(n_failures):
      idx = worst_case_indexes[k]
      test_pred_instance, gt_instance = rescale_bbox(test_url[idx], test_pred[idx], test_y[idx])
      visualize_bounding_box(test_url[idx], test_pred_instance, gt_instance)

    inspect(model, dataset)
    print("Generating grading files...")
    generate_grading_files(model, 'grading_paths.pkl', 'testing_paths.pkl')

if __name__ == '__main__':
    main()
