def extract_misclassified_examples(test_data, y_true, y_pred, num_examples=4):
    misclassified = []
    for idx, (item, true_val, pred_val) in enumerate(zip(test_data, y_true, y_pred)):
        if true_val != pred_val:
            text = item['text']
            lowered = text.lower()
            if "sarcasm" in lowered or "sure, because" in lowered:
                category = "Sarcasm / Irony"
            elif "not" in lowered or "didn't" in lowered:
                category = "Negation / Shift"
            else:
                category = "Long-Range Context"

            misclassified.append({
                "index": idx,
                "text": text,
                "true_label": "Positive" if true_val == 1 else "Negative",
                "pred_label": "Positive" if pred_val == 1 else "Negative",
                "language_pattern": category
            })
            if len(misclassified) >= num_examples:
                break
    return misclassified

def format_master_comparison_table(results_list):
    header = f"{'Model / Experiment':<35} | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'Gap':<6} | {'Time (s)':<8}"
    divider = "-" * len(header)
    rows = [header, divider]

    for r in results_list:
        row = f"{r['model_name']:<35} | {r['accuracy']:<6.4f} | {r['precision']:<6.4f} | {r['recall']:<6.4f} | {r['f1_score']:<6.4f} | {r['train_val_gap']:<6.4f} | {r['training_time_sec']:<8.4f}"
        rows.append(row)

    return "\n".join(rows)
