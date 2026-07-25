import math
import time
from collections import Counter
from src.config import RANDOM_SEED

class PureTFIDFVectorizer:
    """Pure Python TF-IDF Feature Extractor supporting unigrams and bigrams."""
    def __init__(self, ngram_range=(1, 1), max_features=2000):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.vocab = {}
        self.idf = {}

    def _extract_ngrams(self, text):
        words = text.split()
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in range(min_n, max_n + 1):
            for i in range(len(words) - n + 1):
                ngrams.append(" ".join(words[i:i+n]))
        return ngrams

    def fit(self, texts):
        df_counts = Counter()
        total_docs = len(texts)

        for text in texts:
            unique_ngrams = set(self._extract_ngrams(text))
            for ngram in unique_ngrams:
                df_counts[ngram] += 1

        top_ngrams = [ngram for ngram, _ in df_counts.most_common(self.max_features)]
        self.vocab = {ngram: i for i, ngram in enumerate(top_ngrams)}

        for ngram, count in df_counts.items():
            if ngram in self.vocab:
                self.idf[ngram] = math.log((total_docs + 1) / (count + 1)) + 1

    def transform(self, texts):
        vectors = []
        for text in texts:
            ngrams = self._extract_ngrams(text)
            counts = Counter(ngrams)
            vec = [0.0] * len(self.vocab)
            total_words = max(len(ngrams), 1)

            for ngram, count in counts.items():
                if ngram in self.vocab:
                    tf = count / total_words
                    idx = self.vocab[ngram]
                    vec[idx] = tf * self.idf[ngram]
            vectors.append(vec)
        return vectors

class PureLogisticRegression:
    """Pure Python Logistic Regression Classifier using Gradient Descent."""
    def __init__(self, lr=0.1, epochs=30):
        self.lr = lr
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def _sigmoid(self, z):
        return 1.0 / (1.0 + math.exp(-max(min(z, 250), -250)))

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                dot = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                pred = self._sigmoid(dot)
                error = pred - y[i]

                for j in range(n_features):
                    dw[j] += error * X[i][j]
                db += error

            for j in range(n_features):
                self.weights[j] -= (self.lr / n_samples) * dw[j]
            self.bias -= (self.lr / n_samples) * db

    def predict(self, X):
        preds = []
        for row in X:
            dot = sum(row[j] * self.weights[j] for j in range(len(self.weights))) + self.bias
            prob = self._sigmoid(dot)
            preds.append(1 if prob >= 0.5 else 0)
        return preds

class PureSVM:
    """Pure Python Linear Support Vector Machine using Pegasos / SGD."""
    def __init__(self, lr=0.05, lambda_param=0.01, epochs=25):
        self.lr = lr
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        # Convert y {0, 1} to {-1, 1}
        y_signed = [1 if val == 1 else -1 for val in y]

        for _ in range(self.epochs):
            for i in range(n_samples):
                dot = sum(X[i][j] * self.weights[j] for j in range(n_features)) + self.bias
                condition = y_signed[i] * dot >= 1

                for j in range(n_features):
                    if condition:
                        self.weights[j] -= self.lr * (2 * self.lambda_param * self.weights[j])
                    else:
                        self.weights[j] -= self.lr * (2 * self.lambda_param * self.weights[j] - X[i][j] * y_signed[i])
                if not condition:
                    self.bias += self.lr * y_signed[i]

    def predict(self, X):
        preds = []
        for row in X:
            dot = sum(row[j] * self.weights[j] for j in range(len(self.weights))) + self.bias
            preds.append(1 if dot >= 0.0 else 0)
        return preds

def calc_metrics(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)

    total = len(y_true)
    acc = (tp + tn) / total if total > 0 else 0
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    return round(acc, 4), round(p, 4), round(r, 4), round(f1, 4)

def train_and_eval_classical(train_df, val_df, test_df, vectorizer_config=None, model_type="logistic_regression"):
    if vectorizer_config is None:
        vectorizer_config = {"ngram_range": (1, 1), "max_features": 1500}

    vec = PureTFIDFVectorizer(**vectorizer_config)
    train_texts = [d['clean_text'] for d in train_df]
    val_texts = [d['clean_text'] for d in val_df]
    test_texts = [d['clean_text'] for d in test_df]

    y_train = [d['sentiment'] for d in train_df]
    y_val = [d['sentiment'] for d in val_df]
    y_test = [d['sentiment'] for d in test_df]

    vec.fit(train_texts)
    X_train = vec.transform(train_texts)
    X_val = vec.transform(val_texts)
    X_test = vec.transform(test_texts)

    start_time = time.time()
    if model_type == "logistic_regression":
        model = PureLogisticRegression(lr=0.2, epochs=20)
    else:
        model = PureSVM(lr=0.05, lambda_param=0.01, epochs=20)

    model.fit(X_train, y_train)
    train_time = round(time.time() - start_time, 4)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    train_acc, _, _, _ = calc_metrics(y_train, train_preds)
    val_acc, _, _, _ = calc_metrics(y_val, val_preds)
    test_acc, p, r, f1 = calc_metrics(y_test, test_preds)

    return {
        "model_name": f"{model_type.upper()} ({vectorizer_config.get('ngram_range', (1,1))})",
        "vocab_size": len(vec.vocab),
        "accuracy": test_acc,
        "precision": p,
        "recall": r,
        "f1_score": f1,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "train_val_gap": round(abs(train_acc - val_acc), 4),
        "training_time_sec": train_time,
        "test_preds": test_preds
    }
