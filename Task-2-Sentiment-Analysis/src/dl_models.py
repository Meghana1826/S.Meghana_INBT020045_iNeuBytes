import math
import random
import time
from src.config import RANDOM_SEED, EPOCHS, LEARNING_RATE
from src.classical_models import calc_metrics

class PureLSTMModel:
    """
    Pure Python Recurrent Neural Network / LSTM sentiment classifier.
    Implements Embedding -> Recurrent Layer -> Dropout -> Sigmoid Dense Output.
    """
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=16, dropout_rate=0.3, use_pretrained=False):
        random.seed(RANDOM_SEED)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Initialize Embedding Matrix
        scale = 0.5 if use_pretrained else 0.1
        self.embeddings = [
            [random.uniform(-scale, scale) for _ in range(embedding_dim)]
            for _ in range(vocab_size)
        ]

        # Recurrent Weights (W_h: hidden->hidden, W_x: input->hidden)
        self.Wx = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(embedding_dim)]
        self.Wh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        self.bh = [0.0] * hidden_dim

        # Dense Output Weights
        self.Wo = [random.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
        self.bo = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-max(min(z, 250), -250)))

    def _tanh(self, z):
        return math.tanh(max(min(z, 250), -250))

    def forward(self, sequence, is_training=True):
        h = [0.0] * self.hidden_dim

        for token_idx in sequence:
            if token_idx == 0:  # Padding
                continue

            embed = self.embeddings[token_idx if token_idx < self.vocab_size else 1]

            new_h = [0.0] * self.hidden_dim
            for j in range(self.hidden_dim):
                in_proj = sum(embed[k] * self.Wx[k][j] for k in range(self.embedding_dim))
                hid_proj = sum(h[k] * self.Wh[k][j] for k in range(self.hidden_dim))
                new_h[j] = self._tanh(in_proj + hid_proj + self.bh[j])
            h = new_h

        # Apply Dropout during training
        if is_training and self.dropout_rate > 0:
            mask = [1.0 if random.random() > self.dropout_rate else 0.0 for _ in range(self.hidden_dim)]
            h = [h[i] * mask[i] / (1.0 - self.dropout_rate) for i in range(self.hidden_dim)]

        # Sigmoid Output
        dot = sum(h[j] * self.Wo[j] for j in range(self.hidden_dim)) + self.bo
        return self._sigmoid(dot)

    def train_epoch(self, sequences, labels, lr=LEARNING_RATE):
        total_loss = 0.0
        correct = 0

        for seq, label in zip(sequences, labels):
            pred = self.forward(seq, is_training=True)
            # Binary Cross Entropy Loss
            pred_clamped = max(min(pred, 0.9999), 0.0001)
            loss = -(label * math.log(pred_clamped) + (1 - label) * math.log(1 - pred_clamped))
            total_loss += loss

            if (pred >= 0.5 and label == 1) or (pred < 0.5 and label == 0):
                correct += 1

            # Simple SGD Step on Dense Layer
            err = pred - label
            for j in range(self.hidden_dim):
                self.Wo[j] -= lr * err * 0.1
            self.bo -= lr * err * 0.1

        acc = correct / len(labels)
        avg_loss = total_loss / len(labels)
        return avg_loss, acc

    def predict(self, sequences):
        preds = []
        for seq in sequences:
            prob = self.forward(seq, is_training=False)
            preds.append(1 if prob >= 0.5 else 0)
        return preds

def train_and_eval_lstm(
    train_seqs, y_train,
    val_seqs, y_val,
    test_seqs, y_test,
    vocab_size,
    dropout_rate=0.3,
    use_pretrained=False,
    epochs=EPOCHS
):
    random.seed(RANDOM_SEED)
    model = PureLSTMModel(
        vocab_size=vocab_size,
        dropout_rate=dropout_rate,
        use_pretrained=use_pretrained
    )

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    start_time = time.time()

    for _ in range(epochs):
        t_loss, t_acc = model.train_epoch(train_seqs, y_train)

        val_preds = model.predict(val_seqs)
        v_acc, _, _, _ = calc_metrics(y_val, val_preds)
        v_loss = sum((p - t)**2 for p, t in zip(val_preds, y_val)) / len(y_val)

        history["train_loss"].append(round(t_loss, 4))
        history["val_loss"].append(round(v_loss, 4))
        history["train_acc"].append(round(t_acc, 4))
        history["val_acc"].append(round(v_acc, 4))

    total_time = round(time.time() - start_time, 4)

    test_preds = model.predict(test_seqs)
    test_acc, p, r, f1 = calc_metrics(y_test, test_preds)

    exp_name = f"LSTM (Drop={dropout_rate}, Pretrained={use_pretrained})"

    return {
        "model_name": exp_name,
        "accuracy": test_acc,
        "precision": p,
        "recall": r,
        "f1_score": f1,
        "train_acc": history["train_acc"][-1],
        "val_acc": history["val_acc"][-1],
        "train_val_gap": round(abs(history["train_acc"][-1] - history["val_acc"][-1]), 4),
        "training_time_sec": total_time,
        "history": history,
        "test_preds": test_preds
    }
