import os
import random
import math

# Fixed Experimental Controls (Task 2 Requirement)
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Lock random seed across Python for 100% reproducibility."""
    random.seed(seed)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
REPORT_DIR = os.path.join(BASE_DIR, "report")

os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Data Split Config (70% train, 15% val, 15% test - frozen split)
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DATASET_SIZE = 1000  # Clean reproducible review dataset

# Deep Learning / Neural Network Parameters
VOCAB_SIZE = 2000
MAX_SEQ_LEN = 100
EMBEDDING_DIM = 32
HIDDEN_DIM = 32
DEFAULT_DROPOUT = 0.3
EPOCHS = 10
LEARNING_RATE = 0.05
