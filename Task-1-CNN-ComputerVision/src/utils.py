import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tabulate import tabulate
from .config import CLASS_NAMES, OUTPUT_DIR

def get_model_param_count(model):
    """Calculates total trainable parameter count of a Keras model."""
    return sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

def plot_training_history(history, model_name="Baseline"):
    """Plots and saves Loss and Accuracy curves for training vs validation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy Plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='#2b5c8f', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='#d9534f', linewidth=2, linestyle='--')
    axes[0].set_title(f'{model_name} - Accuracy Curve', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss Plot
    axes[1].plot(history.history['loss'], label='Train Loss', color='#2b5c8f', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='#d9534f', linewidth=2, linestyle='--')
    axes[1].set_title(f'{model_name} - Loss Curve', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, f"{model_name}_learning_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Utils] Learning curves saved to: {plot_path}")

def plot_and_analyze_confusion_matrix(y_true, y_pred, model_name="Baseline"):
    """
    Plots confusion matrix heatmaps and identifies top 3 most-confused class pairs.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    cm_path = os.path.join(OUTPUT_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Identify most confused non-diagonal pairs
    cm_off_diag = cm.copy()
    np.fill_diagonal(cm_off_diag, 0)
    
    # Get top 3 misclassifications
    top_indices = np.unravel_index(np.argsort(cm_off_diag.ravel())[::-1][:3], cm_off_diag.shape)
    
    confused_pairs = []
    print(f"\n--- [{model_name}] Top Most Confused Class Pairs ---")
    for i in range(3):
        true_cls = CLASS_NAMES[top_indices[0][i]]
        pred_cls = CLASS_NAMES[top_indices[1][i]]
        count = cm_off_diag[top_indices[0][i], top_indices[1][i]]
        confused_pairs.append((true_cls, pred_cls, count))
        print(f" #{i+1}: True '{true_cls}' misclassified as '{pred_cls}' ({count} instances)")

    return cm, confused_pairs

def evaluate_model_performance(model, x_test, y_test, model_name="Baseline"):
    """
    Computes overall Accuracy, Precision, Recall, F1-score, and outputs classification report.
    """
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test.flatten()

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    test_acc = np.mean(y_true == y_pred)

    print(f"\n================ {model_name} Performance ================")
    print(f" Test Accuracy: {test_acc*100:.2f}%")
    print(f" Macro Precision: {precision:.4f}")
    print(f" Macro Recall:    {recall:.4f}")
    print(f" Macro F1-Score:  {f1:.4f}")
    print("===========================================================")

    cm, confused_pairs = plot_and_analyze_confusion_matrix(y_true, y_pred, model_name=model_name)

    return {
        'model_name': model_name,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confused_pairs': confused_pairs
    }

def print_master_table(experiments_data):
    """Formats and prints Master Experiment Table in Markdown & Tabulate."""
    df = pd.DataFrame(experiments_data)
    table_str = tabulate(df, headers='keys', tablefmt='github', showindex=False)
    print("\n================ Master Experiment Table ================")
    print(table_str)
    print("=========================================================")

    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, "master_experiment_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"[Utils] Master table saved to: {csv_path}")
