# Task 2: Sentiment Analysis using ML and DL - Research Report

**Course ID**: AIINB20726  
**Program**: iNeuBytes Virtual Internship Program (VIIP)  
**Author**: Meghana  
**Date**: July 2026  

---

## 1. Abstract & Research Questions

This study investigates text sentiment classification performance across classical statistical machine learning baselines (Logistic Regression, Support Vector Machine) and deep sequential architectures (Long Short-Term Memory network). Through strict experimental controls (frozen seed = 42, locked 70/15/15 split), we evaluate text representation techniques (TF-IDF unigrams vs. unigrams+bigrams), embedding strategies (trainable vs. pre-trained), and regularization mechanisms (dropout tuning).

### Central Research Questions
1. Does incorporating bigram context in TF-IDF feature extraction significantly improve classical machine learning classification performance?
2. Does an LSTM architecture outperform classical baselines on short-to-medium length movie review texts, and does pre-trained knowledge provide a decisive advantage?
3. How do classical linear decision boundaries compare to recurrent hidden representations when handling negation and sarcasm?

---

## 2. Experimental Setup & Controls

To adhere strictly to fair benchmarking principles, the following controls were locked across all experiments:

| Parameter | Setting / Value | Justification |
|---|---|---|
| **Random Seed** | `42` | Set in Python `random`, `numpy`, and `PyTorch` for 100% reproducibility |
| **Dataset Size** | 5,000 Samples | Balanced positive (50%) and negative (50%) movie reviews |
| **Frozen Split** | Train (70%), Val (15%), Test (15%) | Locked split index shared identically across every baseline & neural model |
| **Text Preprocessing** | Lowercasing, HTML strip, Punctuation & Stopword Removal | Standardized uniform text pipeline |
| **Evaluation Metrics** | Accuracy, Precision, Recall, F1-Score | Equal metric evaluation without cherry-picking |

---

## 3. Part A — Classical Machine Learning Baselines

### Baseline Setup
We trained Logistic Regression and Support Vector Machine (Linear SVM) using TF-IDF vectorization with default unigram settings (`ngram_range=(1,1)`, max features = 5,000).

### Baseline Results
| Model | Vectorizer | Test Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|---|---|---|---|---|---|---|
| **Logistic Regression** | TF-IDF (Unigrams) | **0.9973** | 0.9973 | 0.9973 | **0.9973** | 0.040s |
| **Support Vector Machine (SVM)** | TF-IDF (Unigrams) | **0.9987** | 0.9973 | 1.0000 | **0.9987** | 0.082s |

### Misclassification Hypothesis
Initial error inspection revealed that linear classical models rely heavily on unigram token weights (e.g., `"masterpiece"`, `"dreadful"`). When sentiment depends on negation structures (`"not bad at all"`) or sarcastic phrasing (`"Oh sure, because who doesn't love waiting..."`), unigram models risk misinterpreting individual token polarity without context.

---

## 4. Part B — Controlled Experiments

### 1. Feature Representation Study (Classical)
We varied the n-gram range to evaluate whether bigram context (`unigrams + bigrams`) provides meaningful signal:

- **Logistic Regression (Unigram + Bigram)**: F1 = **0.9987** (improved over 0.9973 unigram baseline).
- **SVM (Unigram + Bigram)**: F1 = **0.9987** (maintained top-tier performance).

*Insight*: Including bigrams captured phrase-level constructs such as `"not bad"` and `"very positive"`, providing sharper decision boundaries.

### 2. Deep Learning Pipeline (LSTM)
Text sequences were tokenized and padded to a maximum sequence length of 200 tokens. The LSTM architecture consists of:
1. `Embedding Layer` (Vocab size = 10,000, Dimension = 128)
2. `LSTM Layer` (Hidden units = 64)
3. `Dropout Layer` (Rate = 0.3)
4. `Dense Layer` + `Sigmoid Activation`

### 3. Embedding & Regularization Study
We conducted controlled experiments isolating embedding type and dropout rates:

| Experiment / Variant | Dropout Rate | Pre-trained Embeddings | Test Accuracy | Test F1-Score | Train-Val Gap | Training Time (s) |
|---|---|---|---|---|---|---|
| **LSTM (Scratch)** | 0.3 | No | **0.9987** | **0.9987** | 0.0013 | 4.85s |
| **LSTM (Pre-trained)** | 0.3 | Yes | 0.9973 | 0.9973 | 0.0020 | 4.92s |
| **LSTM (Zero Dropout)** | 0.0 | No | 0.9987 | 0.9987 | 0.0040 | 4.70s |
| **LSTM (High Dropout)** | 0.5 | No | 0.9973 | 0.9973 | 0.0007 | 4.88s |

*Insight*: Higher dropout (`0.5`) succeeded in minimizing the train-validation generalization gap down to `0.0007` while retaining peak test performance.

---

## 5. Part C — Final Comparison & Evidence-Based Verdict

### Master Comparison Table

| Architecture / Model | Feature Representation | Test Accuracy | Precision | Recall | F1-Score | Training Time |
|---|---|---|---|---|---|---|
| **Support Vector Machine (SVM)** | TF-IDF (Unigram + Bigram) | **0.9987** | 0.9973 | 1.0000 | **0.9987** | **0.082s** |
| **Logistic Regression** | TF-IDF (Unigram + Bigram) | **0.9987** | 0.9973 | 1.0000 | **0.9987** | **0.040s** |
| **LSTM Neural Network** | Learned Embedding (128d) | **0.9987** | 0.9973 | 1.0000 | **0.9987** | 4.850s |

---

## 6. Trade-off Analysis & Final Verdict

### Performance vs. Cost Trade-off
- **Accuracy & F1-Score**: Both SVM (with TF-IDF bigrams) and LSTM achieved matching peak performance (**99.87% F1**).
- **Computational Efficiency**: Logistic Regression and SVM trained in **0.04 - 0.08 seconds**, whereas the LSTM required **~4.85 seconds** (~60x-100x slower on CPU).

### Final Verdict & Deployment Recommendation
> **Recommendation**: Deploy **Linear SVM / Logistic Regression with TF-IDF (Unigrams + Bigrams)** for production sentiment analysis on this dataset.
> 
> **Rationale**: In sentiment benchmark environments where text vocabulary is well-captured by n-grams, classical statistical models deliver equivalent high accuracy at a fraction of the computational overhead, latency, and memory footprint. Deep Learning (LSTM) remains highly valuable for massive, highly variable datasets requiring continuous domain transfer, but classical models prove superior in efficiency-constrained production setups.
