import os
from src.config import ARTIFACTS_DIR

def plot_confusion_matrix(y_true, y_pred, model_name, filename):
    """Generates ASCII/HTML visual confusion matrix representation artifact."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    content = f"""==================================================
CONFUSION MATRIX: {model_name}
==================================================
                 Predicted Negative   Predicted Positive
Actual Negative :      {tn:<14}     {fp:<14}
Actual Positive :      {fn:<14}     {tp:<14}
==================================================
"""
    save_path = os.path.join(ARTIFACTS_DIR, filename.replace('.png', '.txt'))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return save_path

def plot_learning_curves(history, model_name, filename):
    """Generates ASCII learning curve progression artifact."""
    epochs = len(history['train_loss'])
    content = f"LEARNING CURVES PROGRESSION: {model_name}\n" + "="*50 + "\n"
    content += f"{'Epoch':<6} | {'Train Loss':<12} | {'Val Loss':<12} | {'Train Acc':<10} | {'Val Acc':<10}\n"
    content += "-"*50 + "\n"

    for ep in range(epochs):
        t_l = history['train_loss'][ep]
        v_l = history['val_loss'][ep]
        t_a = history['train_acc'][ep]
        v_a = history['val_acc'][ep]
        content += f"{ep+1:<6} | {t_l:<12.4f} | {v_l:<12.4f} | {t_a:<10.4f} | {v_a:<10.4f}\n"

    save_path = os.path.join(ARTIFACTS_DIR, filename.replace('.png', '.txt'))
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return save_path
