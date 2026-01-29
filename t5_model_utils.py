import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


def compute_metrics_t5(tokenizer, label_map):
    def _compute(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if len(predictions.shape) ==3:
            predictions = predictions.argmax(-1)

        predictions = np.clip(predictions, 0, tokenizer.vocab_size -1)
        labels = np.clip(labels, 0, tokenizer.vocab_size-1)

        pred_texts = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

        pred_ids = [label_map.get(p.strip().lower(), -1) for p in pred_texts]
        label_ids = [label_map.get(l.strip().lower(), -1) for l in label_texts]

        acc = accuracy_score(label_ids, pred_ids)
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, pred_ids, average='weighted')
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    return _compute


def t5_classification_report(pred_output, tokenizer, test_df, label_list):
    if isinstance(pred_output, tuple):
        preds = pred_output[0]
    elif hasattr(pred_output, "predictions"):
        preds = pred_output.predictions
    else:
        preds = pred_output

    if hasattr(preds, "ndim") and preds.ndim == 3:
        pred_ids = preds.argmax(-1)
    else:
        pred_ids = preds

    if isinstance(pred_ids, np.ndarray):
        pred_ids = pred_ids.tolist()

    cleaned_pred_ids = []
    for seq in pred_ids:
        if isinstance(seq, list) and all(isinstance(tok, int) for tok in seq):
            cleaned_pred_ids.append(seq)
        elif isinstance(seq, list) and len(seq) == 1 and isinstance(seq[0], list):
            cleaned_pred_ids.append(seq[0])
        else:
            cleaned_pred_ids.append([tok for tok in seq if isinstance(tok, int)])

    pred_texts = tokenizer.batch_decode(cleaned_pred_ids, skip_special_tokens=True)
    pred_texts = [t.strip().lower() for t in pred_texts]
    references = [l.strip().lower() for l in test_df["target_text"].tolist()]
    label_list_lower = [l.lower() for l in label_list]

    filtered = [(r, p) for r, p in zip(references, pred_texts)
                if r in label_list_lower and p in label_list_lower]
    if not filtered:
        raise ValueError("No valid labels remain after filtering.")
    references, pred_texts = zip(*filtered)

    return classification_report(
        references,
        pred_texts,
        labels=label_list_lower,
        target_names=label_list_lower,
        digits=4,
        zero_division=0,
    )
