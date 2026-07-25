import os
import random
import numpy as np
import tensorflow as tf

# Global Seed for Reproducibility
SEED = 42

def set_global_seed(seed=SEED):
    """Locks the random seed across Python, NumPy, and TensorFlow."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"[Config] Global random seed set to: {seed}")

# Dataset Configuration
NUM_CLASSES = 10
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
VAL_SPLIT_RATIO = 0.2  # 80% Train (40,000), 20% Validation (10,000)

# Training Hyperparameters
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_OPTIMIZER = 'adam'

# Directories
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)
