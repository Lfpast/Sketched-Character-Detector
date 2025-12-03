
#!/bin/bash

# Minimal setup + GPU check
ENV_NAME="2211-pa2"

# Check conda
if ! command -v conda &>/dev/null; then
  echo "Error: conda is not installed. Please install Miniconda/Anaconda first."
  exit 1
fi

eval "$(conda shell.bash hook)"

# Create or update env from environment.yml
if conda env list | grep -q "^$ENV_NAME\s"; then
  echo "Updating existing environment: $ENV_NAME"
  if ! conda env update -n "$ENV_NAME" -f environment.yml --prune; then
    echo "Conda environment update from environment.yml failed. Falling back to explicit conda install command..."
    fallback_failed=0
  fi
else
  echo "Creating environment: $ENV_NAME"
  if ! conda env create -f environment.yml; then
    echo "Conda environment creation from environment.yml failed. Falling back to explicit conda install command..."
    fallback_failed=0
  fi
fi

# If fallback_needed, attempt a direct conda create with essential packages
if [ "${fallback_failed:-1}" = "0" ]; then
  echo "Attempting fallback conda create with tensorflow and base packages..."
  # remove existing env to avoid conflicts (only if it exists)
  if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Removing partially created environment $ENV_NAME"
    conda env remove -n "$ENV_NAME" -y || true
  fi
  if ! conda create -n "$ENV_NAME" -c conda-forge python=3.10 tensorflow=2.16 numpy=1.25 matplotlib opencv pip -y; then
    echo "Fallback conda create failed. Please inspect the error message above for details and try installing manually with conda or use a system admin to install drivers." >&2
    exit 1
  fi
fi

echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Install optional pip packages if requirements.txt exists
if [ -f requirements.txt ]; then
  echo "Installing requirements from requirements.txt"
  pip install -r requirements.txt
fi

echo "Checking NVIDIA ability and TensorFlow GPU..."
if command -v nvidia-smi &>/dev/null; then
  echo "nvidia-smi found - driver info:";
  nvidia-smi
else
  echo "nvidia-smi not found. NVIDIA drivers may be missing (or you're not on a GPU host)."
fi

echo "Setup and checks finished. If GPUs are not visible, ensure system NVIDIA driver + CUDA are installed and that cuda/cuDNN versions match the package versions (see environment.yml)."
