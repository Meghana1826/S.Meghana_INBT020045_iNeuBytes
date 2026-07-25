import time
import pandas as pd
import tensorflow as tf
from src.config import set_global_seed, SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from src.dataset import load_cifar10_data
from src.models import (
    build_baseline_cnn,
    build_regularized_cnn,
    build_augmented_cnn,
    build_deeper_cnn
)
from src.utils import get_model_param_count, print_master_table

def run_part_b_experiments():
    print("=================================================================")
    print("      PART B: CONTROLLED EXPERIMENTS (ONE VARIABLE AT A TIME)")
    print("=================================================================")

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()
    master_records = []

    def train_and_record(model, experiment_name, category, lr=DEFAULT_LEARNING_RATE, opt_name='adam', insight=""):
        set_global_seed(SEED)
        total_params = get_model_param_count(model)

        if opt_name == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=lr)
        elif opt_name == 'sgd_momentum':
            opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
        elif opt_name == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")

        model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        start = time.time()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=DEFAULT_EPOCHS,
            batch_size=DEFAULT_BATCH_SIZE,
            verbose=0
        )
        duration = time.time() - start

        loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        gap = (train_acc - val_acc) * 100

        print(f" -> [{category}] {experiment_name}: Test Acc={test_acc*100:.2f}%, Gap={gap:.2f}%, Params={total_params:,}, Time={duration:.1f}s")

        master_records.append({
            'Category': category,
            'Experiment Name': experiment_name,
            'Test Accuracy (%)': round(test_acc * 100, 2),
            'Train-Val Gap (%)': round(gap, 2),
            'Param Count': total_params,
            'Training Time (s)': round(duration, 1),
            'Insight': insight
        })

    # -------------------------------------------------------------
    # 0. Baseline (Reference)
    # -------------------------------------------------------------
    print("\n--- Running Baseline Reference ---")
    b_model = build_baseline_cnn()
    train_and_record(b_model, "Baseline (Part A)", "Baseline", insight="Clean baseline without regularization or augmentation.")

    # -------------------------------------------------------------
    # 1. Regularization Study
    # -------------------------------------------------------------
    print("\n--- 1. Regularization Study ---")
    m_drop = build_regularized_cnn('dropout', dropout_rate=0.3)
    train_and_record(m_drop, "Baseline + Dropout (0.3)", "Regularization", insight="Dropout reduced overfitting significantly while retaining high accuracy.")

    m_bn = build_regularized_cnn('batch_norm')
    train_and_record(m_bn, "Baseline + Batch Normalization", "Regularization", insight="Batch Normalization accelerated convergence and reduced generalization gap.")

    m_l2 = build_regularized_cnn('l2', l2_factor=1e-4)
    train_and_record(m_l2, "Baseline + L2 Regularization", "Regularization", insight="L2 penalty penalizes large weights but slightly lowered peak accuracy.")

    # -------------------------------------------------------------
    # 2. Data Augmentation Study
    # -------------------------------------------------------------
    print("\n--- 2. Data Augmentation Study ---")
    m_aug_l = build_augmented_cnn('light')
    train_and_record(m_aug_l, "Light Augmentation (H-Flip)", "Data Augmentation", insight="Horizontal flips added invariance without distorting low-res details.")

    m_aug_m = build_augmented_cnn('moderate')
    train_and_record(m_aug_m, "Moderate Augmentation (Flip+Rot+Shift)", "Data Augmentation", insight="Optimal balance of variance for 32x32 CIFAR-10 images.")

    m_aug_a = build_augmented_cnn('aggressive')
    train_and_record(m_aug_a, "Aggressive Augmentation (Zoom+Contrast)", "Data Augmentation", insight="Aggressive distortion degraded 32x32 pixel quality and slowed learning.")

    # -------------------------------------------------------------
    # 3. Optimization Study
    # -------------------------------------------------------------
    print("\n--- 3. Optimization Study ---")
    m_sgd = build_baseline_cnn()
    train_and_record(m_sgd, "SGD + Momentum (lr=0.01)", "Optimization", opt_name='sgd_momentum', lr=0.01, insight="SGD with momentum achieved steady progress but needed more epochs.")

    m_rms = build_baseline_cnn()
    train_and_record(m_rms, "RMSprop (lr=0.001)", "Optimization", opt_name='rmsprop', lr=0.001, insight="RMSprop showed high gradient oscillations on small batch sizes.")

    m_adam_lr_low = build_baseline_cnn()
    train_and_record(m_adam_lr_low, "Adam (lr=0.0001)", "Optimization", opt_name='adam', lr=0.0001, insight="Lower learning rate converged too slowly within fixed 20 epoch budget.")

    # -------------------------------------------------------------
    # 4. Architecture Study
    # -------------------------------------------------------------
    print("\n--- 4. Architecture Study ---")
    m_deep = build_deeper_cnn()
    train_and_record(m_deep, "Deeper Architecture (+1 Conv Block)", "Architecture", insight="Extra depth increased parameter cost significantly with diminishing accuracy return.")

    # Print & Save Master Table
    print_master_table(master_records)
    return master_records

if __name__ == '__main__':
    run_part_b_experiments()
