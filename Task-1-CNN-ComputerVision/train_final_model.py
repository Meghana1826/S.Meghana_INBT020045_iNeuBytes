import time
import pandas as pd
import numpy as np
import tensorflow as tf
from src.config import set_global_seed, SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from src.dataset import load_cifar10_data
from src.models import build_baseline_cnn, build_customized_cnn
from src.utils import evaluate_model_performance, plot_training_history, get_model_param_count

def run_part_c_final_model():
    print("=================================================================")
    print("      PART C: FINAL CUSTOMIZED CNN (WINNING COMBINATION)")
    print("=================================================================")

    set_global_seed(SEED)
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()

    # 1. Train Baseline for head-to-head comparison
    print("\n--- Training Baseline CNN for Direct Head-to-Head Comparison ---")
    baseline_model = build_baseline_cnn()
    baseline_params = get_model_param_count(baseline_model)
    baseline_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    b_start = time.time()
    b_history = baseline_model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, verbose=0
    )
    b_time = time.time() - b_start
    b_results = evaluate_model_performance(baseline_model, x_test, y_test, model_name="PartA_Baseline")

    # 2. Build & Train Final Customized CNN
    print("\n--- Training Final Customized CNN ---")
    custom_model = build_customized_cnn(aug_level='moderate')
    custom_model.summary()

    custom_params = get_model_param_count(custom_model)
    custom_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    c_start = time.time()
    c_history = custom_model.fit(
        x_train, y_train, validation_data=(x_val, y_val),
        epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, verbose=1
    )
    c_time = time.time() - c_start

    # Plot customized curves
    plot_training_history(c_history, model_name="PartC_Customized_CNN")

    # Evaluate customized model
    c_results = evaluate_model_performance(custom_model, x_test, y_test, model_name="PartC_Customized_CNN")

    # 3. Comparative Trade-off Analysis
    b_acc = b_results['test_acc'] * 100
    c_acc = c_results['test_acc'] * 100
    acc_improvement = c_acc - b_acc

    param_diff_millions = (custom_params - baseline_params) / 1e6
    acc_per_million_params = acc_improvement / param_diff_millions if param_diff_millions != 0 else 0

    print("\n================ FINAL COMPARISON SUMMARY ================")
    print(f" Baseline Test Accuracy:     {b_acc:.2f}%")
    print(f" Customized Test Accuracy:   {c_acc:.2f}%")
    print(f" Absolute Accuracy Gain:     +{acc_improvement:.2f}% (Target: >= +3.0%)")
    print(f" Baseline Params:           {baseline_params:,}")
    print(f" Customized Params:         {custom_params:,}")
    print(f" Accuracy Gained / Million Params: {acc_per_million_params:.2f}%")
    print(f" Baseline Training Time:    {b_time:.1f}s")
    print(f" Customized Training Time:  {c_time:.1f}s")
    print(f" Threshold Check (>= +3.0%): {'PASSED SUCCESS THRESHOLD!' if acc_improvement >= 3.0 else 'CHECK HYPERPARAMETERS'}")
    print("=========================================================")

    # Confusion pair comparison
    print("\n--- Confusion Analysis Comparison ---")
    print("Part A Baseline Top Confused Pairs:", b_results['confused_pairs'])
    print("Part C Final Model Top Confused Pairs:", c_results['confused_pairs'])

    return {
        'baseline_results': b_results,
        'custom_results': c_results,
        'acc_improvement': acc_improvement,
        'acc_per_million_params': acc_per_million_params
    }

if __name__ == '__main__':
    run_part_c_final_model()
