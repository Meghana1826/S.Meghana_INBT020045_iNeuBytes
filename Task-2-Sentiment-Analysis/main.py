from src.config import set_seed, RANDOM_SEED, MAX_SEQ_LEN, VOCAB_SIZE
from src.data_loader import load_sentiment_dataset, get_frozen_splits
from src.preprocessing import clean_text, CustomTokenizer
from src.classical_models import train_and_eval_classical
from src.dl_models import train_and_eval_lstm
from src.evaluator import extract_misclassified_examples, format_master_comparison_table
from src.visualizer import plot_confusion_matrix, plot_learning_curves

def main():
    print("=" * 80)
    print("TASK 2: SENTIMENT ANALYSIS USING ML AND DL - EXPERIMENTAL PIPELINE")
    print("=" * 80)

    # 1. Lock seed
    set_seed(RANDOM_SEED)
    print(f"[1/6] Random seed locked to: {RANDOM_SEED}")

    # 2. Load dataset and lock frozen split
    print("[2/6] Loading benchmark sentiment dataset and freezing splits...")
    dataset = load_sentiment_dataset()
    print(f"Total dataset samples: {len(dataset)}")

    train_data, val_data, test_data = get_frozen_splits(dataset)
    print(f"Frozen Split locked: Train={len(train_data)} | Val={len(val_data)} | Test={len(test_data)}")

    # 3. Clean text
    print("[3/6] Applying text preprocessing (lowercasing, punctuation & stopword removal)...")
    for item in train_data:
        item['clean_text'] = clean_text(item['text'])
    for item in val_data:
        item['clean_text'] = clean_text(item['text'])
    for item in test_data:
        item['clean_text'] = clean_text(item['text'])

    all_results = []

    # ---------------------------------------------------------
    # PART A: Classical ML Baselines
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("PART A: Classical Machine Learning Baselines")
    print("-" * 50)

    logreg_res = train_and_eval_classical(
        train_data, val_data, test_data,
        vectorizer_config={"ngram_range": (1, 1), "max_features": 1500},
        model_type="logistic_regression"
    )
    all_results.append(logreg_res)
    plot_confusion_matrix([d['sentiment'] for d in test_data], logreg_res['test_preds'], "Logistic Regression", "confusion_matrix_logistic_regression.png")
    print(f"Logistic Regression (Unigrams) -> Test Acc: {logreg_res['accuracy']} | F1: {logreg_res['f1_score']}")

    svm_res = train_and_eval_classical(
        train_data, val_data, test_data,
        vectorizer_config={"ngram_range": (1, 1), "max_features": 1500},
        model_type="svm"
    )
    all_results.append(svm_res)
    plot_confusion_matrix([d['sentiment'] for d in test_data], svm_res['test_preds'], "SVM (Linear)", "confusion_matrix_svm.png")
    print(f"SVM (Unigrams) -> Test Acc: {svm_res['accuracy']} | F1: {svm_res['f1_score']}")

    # ---------------------------------------------------------
    # PART B: Controlled Experiments
    # ---------------------------------------------------------
    print("\n" + "-" * 50)
    print("PART B: Controlled Experiments (Representation & Architecture)")
    print("-" * 50)

    # 1. Feature representation study (Unigrams vs Unigrams+Bigrams)
    logreg_bigram = train_and_eval_classical(
        train_data, val_data, test_data,
        vectorizer_config={"ngram_range": (1, 2), "max_features": 1500},
        model_type="logistic_regression"
    )
    all_results.append(logreg_bigram)
    print(f"Logistic Regression (Unigram+Bigram) -> Test Acc: {logreg_bigram['accuracy']} | F1: {logreg_bigram['f1_score']}")

    svm_bigram = train_and_eval_classical(
        train_data, val_data, test_data,
        vectorizer_config={"ngram_range": (1, 2), "max_features": 1500},
        model_type="svm"
    )
    all_results.append(svm_bigram)
    print(f"SVM (Unigram+Bigram) -> Test Acc: {svm_bigram['accuracy']} | F1: {svm_bigram['f1_score']}")

    # 2. Tokenize & Pad Sequences for Deep Learning
    print("\nTokenizing sequences for Deep Learning LSTM models...")
    tokenizer = CustomTokenizer(vocab_size=VOCAB_SIZE, max_len=MAX_SEQ_LEN)
    tokenizer.fit_on_texts([d['clean_text'] for d in train_data])

    train_seqs = [tokenizer.text_to_sequence(d['clean_text']) for d in train_data]
    val_seqs = [tokenizer.text_to_sequence(d['clean_text']) for d in val_data]
    test_seqs = [tokenizer.text_to_sequence(d['clean_text']) for d in test_data]

    y_train = [d['sentiment'] for d in train_data]
    y_val = [d['sentiment'] for d in val_data]
    y_test = [d['sentiment'] for d in test_data]

    # 3. Embedding Study
    print("\nRunning Embedding Study (Scratch vs Pretrained)...")
    lstm_scratch = train_and_eval_lstm(
        train_seqs, y_train, val_seqs, y_val, test_seqs, y_test,
        vocab_size=len(tokenizer.word2idx), dropout_rate=0.3, use_pretrained=False
    )
    all_results.append(lstm_scratch)
    print(f"LSTM (Scratch Embedding) -> Test Acc: {lstm_scratch['accuracy']} | F1: {lstm_scratch['f1_score']}")

    lstm_pretrained = train_and_eval_lstm(
        train_seqs, y_train, val_seqs, y_val, test_seqs, y_test,
        vocab_size=len(tokenizer.word2idx), dropout_rate=0.3, use_pretrained=True
    )
    all_results.append(lstm_pretrained)
    print(f"LSTM (Pretrained Embedding) -> Test Acc: {lstm_pretrained['accuracy']} | F1: {lstm_pretrained['f1_score']}")

    # 4. Regularization Study
    print("\nRunning Regularization Study (Dropout 0.0 vs 0.5)...")
    lstm_drop0 = train_and_eval_lstm(
        train_seqs, y_train, val_seqs, y_val, test_seqs, y_test,
        vocab_size=len(tokenizer.word2idx), dropout_rate=0.0, use_pretrained=False
    )
    all_results.append(lstm_drop0)

    lstm_drop5 = train_and_eval_lstm(
        train_seqs, y_train, val_seqs, y_val, test_seqs, y_test,
        vocab_size=len(tokenizer.word2idx), dropout_rate=0.5, use_pretrained=False
    )
    all_results.append(lstm_drop5)

    # Save Best LSTM Artifacts
    best_lstm = max([lstm_scratch, lstm_pretrained, lstm_drop0, lstm_drop5], key=lambda x: x['f1_score'])
    plot_learning_curves(best_lstm['history'], "Best LSTM", "lstm_learning_curves.png")
    plot_confusion_matrix(y_test, best_lstm['test_preds'], "Best LSTM", "confusion_matrix_lstm.png")

    # ---------------------------------------------------------
    # PART C: Final Comparison & Synthesis
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("PART C: MASTER COMPARISON TABLE & ERROR ANALYSIS")
    print("=" * 80)

    master_table = format_master_comparison_table(all_results)
    print(master_table)

    # Error Analysis
    print("\n--- Misclassification Error Inspection ---")
    y_test_vals = [d['sentiment'] for d in test_data]
    misclassified_lr = extract_misclassified_examples(test_data, y_test_vals, logreg_res['test_preds'], num_examples=3)
    if not misclassified_lr:
        print("Zero misclassifications detected on test split (Perfect baseline alignment!).")
    else:
        for idx, ex in enumerate(misclassified_lr, 1):
            print(f"[{idx}] Category: {ex['language_pattern']} | True: {ex['true_label']} | Pred: {ex['pred_label']}")
            print(f"    Text snippet: {ex['text'][:100]}...\n")

    print("=" * 80)
    print("Pipeline Execution Completed Successfully! Artifacts saved in /artifacts")
    print("=" * 80)

if __name__ == "__main__":
    main()
