import tensorflow as tf
from sklearn.model_selection import train_test_split
from .config import SEED, VAL_SPLIT_RATIO, set_global_seed

def load_cifar10_data():
    """
    Loads CIFAR-10 dataset, normalizes pixel values to [0, 1],
    and creates a frozen train/validation/test split.
    """
    set_global_seed(SEED)
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize pixel values
    x_train_full = x_train_full.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Frozen Train/Val Split using locked seed
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full,
        test_size=VAL_SPLIT_RATIO,
        random_state=SEED,
        stratify=y_train_full
    )

    print(f"[Dataset] Train samples: {x_train.shape[0]}")
    print(f"[Dataset] Validation samples: {x_val.shape[0]}")
    print(f"[Dataset] Test samples: {x_test.shape[0]}")

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def get_augmentation_layer(level='light'):
    """
    Returns Keras Data Augmentation sequential layer based on study level.
    - 'light': Horizontal Flip
    - 'moderate': Flip + Small Rotation (0.1) & Width/Height Shift (0.1)
    - 'aggressive': Flip + Rotation (0.2) + Zoom (0.2) + Contrast Adjustment
    """
    if level == 'light':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=SEED)
        ], name="light_augmentation")
    elif level == 'moderate':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.1, seed=SEED),
            tf.keras.layers.RandomTranslation(0.1, 0.1, seed=SEED)
        ], name="moderate_augmentation")
    elif level == 'aggressive':
        return tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.2, seed=SEED),
            tf.keras.layers.RandomZoom(0.2, seed=SEED),
            tf.keras.layers.RandomContrast(0.2, seed=SEED)
        ], name="aggressive_augmentation")
    else:
        raise ValueError(f"Unknown augmentation level: {level}")
