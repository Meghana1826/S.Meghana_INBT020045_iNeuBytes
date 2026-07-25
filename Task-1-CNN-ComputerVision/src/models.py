import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from .dataset import get_augmentation_layer

def build_baseline_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Part A: Baseline CNN (AlexNet-style adapted for 32x32 images).
    Clean baseline: No Data Augmentation, No Batch Normalization, No Dropout.
    Filter progression: 64 -> 128 -> 256.
    """
    model = models.Sequential(name="Baseline_CNN")

    # Block 1: Conv 64 filters
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 2: Conv 128 filters
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Block 3: Conv 256 filters
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Classification Head
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_regularized_cnn(reg_type='dropout', dropout_rate=0.3, l2_factor=1e-4, input_shape=(32, 32, 3), num_classes=10):
    """
    Part B.1: Regularization Study Models
    reg_type options: 'dropout', 'batch_norm', 'l2'
    """
    model = models.Sequential(name=f"CNN_Reg_{reg_type}")
    kernel_reg = regularizers.l2(l2_factor) if reg_type == 'l2' else None

    # Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_reg, input_shape=input_shape))
    if reg_type == 'batch_norm':
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    if reg_type == 'dropout':
        model.add(layers.Dropout(dropout_rate))

    # Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_reg))
    if reg_type == 'batch_norm':
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    if reg_type == 'dropout':
        model.add(layers.Dropout(dropout_rate))

    # Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=kernel_reg))
    if reg_type == 'batch_norm':
        model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D((2, 2)))
    if reg_type == 'dropout':
        model.add(layers.Dropout(dropout_rate))

    # Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu', kernel_regularizer=kernel_reg))
    if reg_type == 'dropout':
        model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(128, activation='relu', kernel_regularizer=kernel_reg))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_augmented_cnn(aug_level='light', input_shape=(32, 32, 3), num_classes=10):
    """
    Part B.2: Data Augmentation Study Models
    """
    model = models.Sequential(name=f"CNN_Aug_{aug_level}")
    aug_layer = get_augmentation_layer(aug_level)
    model.add(aug_layer)

    # Standard baseline architecture
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_deeper_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Part B.4: Architecture Study Model (Adds 4th Conv block: 512 filters).
    """
    model = models.Sequential(name="CNN_Deeper_Architecture")

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Extra Conv Block
    model.add(layers.Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model


def build_customized_cnn(aug_level='moderate', input_shape=(32, 32, 3), num_classes=10):
    """
    Part C: Final Customized CNN.
    Combines verified winning techniques:
    - Moderate Data Augmentation
    - Batch Normalization + Spatial Dropout
    - Increasing filters: 64 -> 128 -> 256
    """
    model = models.Sequential(name="Final_Customized_CNN")

    # Data Augmentation
    model.add(get_augmentation_layer(aug_level))

    # Conv Block 1
    model.add(layers.Conv2D(64, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Conv Block 2
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.3))

    # Conv Block 3
    model.add(layers.Conv2D(256, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.4))

    # Dense Head
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
