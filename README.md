# Sketched Character Detector with ResNet50

This project implements an object detection model using a pre-trained ResNet50 architecture. The model is designed to predict bounding boxes for specific objects within images.

## Project Structure

- **data/**: Contains data loading and preprocessing scripts.
- **model/**: Defines the neural network architecture and loss functions.
- **train/**: Handles the training loop and model saving.
- **evaluate/**: Contains evaluation metrics and failure case analysis.
- **utils/**: Utility functions for visualization and other tasks.
- **main.py**: The entry point for training and evaluating the model.

## Setup

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate project-env
   ```

2. Ensure your data is placed in the `data/` directory.

## Usage

To train and evaluate the model, run:

```bash
python main.py
```

The script will:
1. Load and preprocess the dataset.
2. Compile the ResNet50-based model.
3. Train the model on the training set.
4. Export trained parameters.
5. Evaluate the model on the test set.
6. Extract and visualize failure cases.
7. Generate prediction files for grading and testing.

## Model Details

The model uses a ResNet50 backbone (pre-trained on ImageNet) with custom top layers for regression. The output is a bounding box coordinate set (x, y, width, height). The loss function used is GIoU (Generalized Intersection over Union) or MSE, optimized using RMSprop.
