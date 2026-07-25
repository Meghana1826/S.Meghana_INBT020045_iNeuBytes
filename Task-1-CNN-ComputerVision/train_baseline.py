import time
import os
import tensorflow as tf
from src.config import set_global_seed, SEED, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE
from src.dataset import load_cifar10_data
from src.models import build_baseline_cnn
from src.utils import evaluate_model_performance, plot_training_history, get_model_param_count

def run_part_a_baseline():
    print("=================================================================")
    print("      PART A: TRADITIONAL CNN MODEL (THE BASELINE IMPLEMENTATION)")
    print("=================================================================")
    set_global_seed(SEED)

    # 1. Load Data
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_cifar10_data()

    # 2. Build Model
    model = build_baseline_cnn()
    model.summary()

    total_params = get_model_param_count(model)
    print(f"\n[Part A] Total Model Parameters: {total_params:,}")

    # 3. Compile Model
    optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 4. Train Model
    start_time = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=DEFAULT_EPOCHS,
        batch_size=DEFAULT_BATCH_SIZE,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"\n[Part A] Baseline Training Completed in {training_time:.2f} seconds.")

    # 5. Save Learning Curves
    plot_training_history(history, model_name="PartA_Baseline_CNN")

    # 6. Evaluate Baseline Model
    eval_results = evaluate_model_performance(model, x_test, y_test, model_name="PartA_Baseline_CNN")

    train_acc = history.history['accuracy'][-1]
    val_acc = history.history['val_accuracy'][-1]
    train_val_gap = (train_acc - val_acc) * 100

    print(f"\n[Part A Summary]")
    print(f" Train Accuracy: {train_acc*100:.2f}%")
    print(f" Val Accuracy:   {val_acc*100:.2f}%")
    print(f" Train-Val Gap:  {train_val_gap:.2f}%")
    print(f" Test Accuracy:  {eval_results['test_acc']*100:.2f}%")
    print(f" Success Threshold (>= 70%): {'PASSED' if eval_results['test_acc'] >= 0.70 else 'FAILED'}")

    return {
        'model_name': 'Part A Baseline CNN',
        'test_acc': round(eval_results['test_acc'] * 100, 2),
        'train_val_gap': round(train_val_gap, 2),
        'param_count': total_params,
        'training_time_sec': round(training_time, 2),
        'history': history,
        'eval_results': eval_results
    }

if __name__ == '__main__':
    run_part_a_baseline()
