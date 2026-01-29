# train.py
import argparse
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from subprocess import run
from datetime import datetime
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import csv
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
from model import debert_model, xlnet_model, t5_model
from t5_model_utils import t5_classification_report


MODEL_REGISTRY = {
    "deberta": debert_model,
    "xlnet": xlnet_model,
    "t5": t5_model
}

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = ""


def load_and_split_dataset(dataset_name):
    dataset_path = f"dataset/{dataset_name}.csv"
    train_path = f"dataset/{dataset_name}_train.csv"
    test_path = f"dataset/{dataset_name}_test.csv"

    if os.path.exists(train_path) and os.path.exists(test_path):
        train_df = pd.read_csv(train_path, encoding='windows-1252', quotechar='"')
        test_df = pd.read_csv(test_path, encoding='windows-1252')
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    else:
        df = pd.read_csv(dataset_path, encoding='windows-1252', quotechar='"')
        if dataset_name == "github":
            valid_labels = ["positive", "neutral", "negative"]
            df = df[df['label'].str.lower().isin(valid_labels)]
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)

    return train_df, val_df, test_df


def apply_da_script(dataset_name, da_method, train_df):
    if da_method == "none":
        return train_df
    input_path = f"temp/{dataset_name}_orig.csv"
    output_path = f"temp/{dataset_name}_{da_method}.csv"
    if da_method == "github":
        train_df.to_csv(input_path, index=False, quoting=csv.QUOTE_ALL)
    else:
        train_df.to_csv(input_path, index=False)
    run(["python", f"augment/{da_method}.py", "--input", input_path, "--output", output_path], env=env)
    return pd.read_csv(output_path)


def tokenize_function(example, tokenizer, model_key):
    if model_key == "t5":
        input_text = example.get("input_text", "")
        target_text = str(example.get("target_text", ""))

        model_input = tokenizer(
            text=input_text,
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                text=target_text,
                truncation=True,
                padding="max_length",
                max_length=64,
            )

        model_input["labels"] = labels["input_ids"]
        return model_input
    else:
        tokenized = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=256,
        )
        if "label_vector" in example:
            tokenized["labels"] = torch.tensor(example["label_vector"], dtype=torch.float32)
        else:
            tokenized["labels"] = int(example["label"])
        return tokenized


def train_model(model_key, train_df, val_df, test_df, output_file, is_multilabel=False):
    if is_multilabel:
        label_list = sorted(
            set(
                ",".join(
                    pd.concat([train_df["label"], val_df["label"], test_df["label"]])
                    .dropna().astype(str).tolist()
                ).split(",")
            )
        )
        mlb = MultiLabelBinarizer(classes=label_list)
        mlb.fit([label_list])

        def encode(df):
            df = df.dropna(subset=["text", "label"]).copy()
            df["label"] = df["label"].apply(lambda x: x.split(",") if isinstance(x, str) else [])
            if model_key == "t5":
                df["input_text"] = "classify: " + df["text"]
                df["target_text"] = df["label"].apply(lambda labels: ",".join(labels))
            else:
                df["label_vector"] = mlb.transform(df["label"]).tolist()
            return df

        train_df = encode(train_df)
        val_df = encode(val_df)
        test_df = encode(test_df)
    else:
        label_list = sorted(
            pd.concat([train_df["label"], val_df["label"], test_df["label"]])
            .dropna().unique()
        )
        label_map = {v: i for i, v in enumerate(label_list)}

        def encode_single(df):
            df = df.dropna(subset=["text", "label"]).copy()
            if model_key == "t5":
                df["input_text"] = "classify: " + df["text"]
                df["target_text"] = df["label"].astype(str).str.lower()
            else:
                df["label"] = df["label"].map(label_map)
                df = df.dropna(subset=["label"])
                df["label"] = df["label"].astype(int)
            return df

        train_df = encode_single(train_df)
        val_df = encode_single(val_df)
        test_df = encode_single(test_df)

    if is_multilabel:
        tokenizer, model = MODEL_REGISTRY[model_key].load_model_multi(num_labels=len(label_list))
    else:
        tokenizer, model = MODEL_REGISTRY[model_key].load_model(num_labels=len(label_list))

    tokenized_train = Dataset.from_pandas(train_df).map(
        lambda x: tokenize_function(x, tokenizer, model_key),
        remove_columns=train_df.columns.tolist(),
    )
    tokenized_val = Dataset.from_pandas(val_df).map(
        lambda x: tokenize_function(x, tokenizer, model_key),
        remove_columns=val_df.columns.tolist(),
    )
    tokenized_test = Dataset.from_pandas(test_df).map(
        lambda x: tokenize_function(x, tokenizer, model_key),
        remove_columns=test_df.columns.tolist(),
    )

    def compute_metrics_t5(tokenizer, label_map):
        def _compute(eval_pred):
            predictions, labels = eval_pred

            if isinstance(predictions, tuple):
                predictions = predictions[0]
            if len(predictions.shape) ==3:
                predictions = predictions.argmax(-1)

            vocab_size = tokenizer.vocab_size
            predictions = np.clip(predictions, 0, vocab_size-1)
            labels = np.clip(labels, 0, vocab_size-1)

            predictions = predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            labels = labels.tolist() if isinstance(labels, np.ndarray) else labels

            if isinstance(predictions[0], list):
                predictions = [p[0] if isinstance(p, list) else p for p in predictions]
            if isinstance(labels[0], list):
                labels = [l[0] if isinstance(l, list) else l for l in labels]


            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [label_map.get(p.strip().lower(), -1) for p in decoded_preds]
            decoded_labels = [label_map.get(l.strip().lower(), -1) for l in decoded_labels]

            acc = accuracy_score(decoded_labels, decoded_preds)
            precision, recall, f1, _ = precision_recall_fscore_support(decoded_labels, decoded_preds, average='weighted', zero_division=0)
            return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
        return _compute

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        decoded_preds = predictions.argmax(-1)
        decoded_labels = labels

        if isinstance(decoded_preds[0], (list, np.ndarray)):
            decoded_preds = [x[0] for x in decoded_preds]
        if isinstance(decoded_labels[0], (list, np.ndarray)):
            decoded_labels = [x[0] for x in decoded_labels]

        acc = accuracy_score(decoded_labels, decoded_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(decoded_labels, decoded_preds, average='weighted', zero_division=0)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    def compute_metrics_t5_multilabel(label_list, tokenizer):
        mlb = MultiLabelBinarizer(classes=label_list)
        mlb.fit([label_list])

        def _compute(eval_pred):
            predictions, labels = eval_pred

            if isinstance(predictions, tuple):
                predictions = predictions[0]

            decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            decoded_preds = [p.strip().lower().split(",") for p in decoded_preds]
            decoded_labels = [l.strip().lower().split(",") for l in decoded_labels]

            y_pred = mlb.transform(decoded_preds)
            y_true = mlb.transform(decoded_labels)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )
            return {"precision": precision, "recall": recall, "f1": f1}

        return _compute

    def compute_metrics_multilabel(label_list):
        def _compute(eval_pred):
            predictions, labels = eval_pred

            probs = torch.sigmoid(torch.tensor(predictions)).numpy()
            preds = (probs >= 0.5).astype(int)
            labels = np.array(labels)

            precision, recall, f1, _ = precision_recall_fscore_support(
                labels, preds, average="macro", zero_division=0
            )
            return {"precision": precision, "recall": recall, "f1": f1}

        return _compute

    if model_key == "t5":
        if is_multilabel:
            compute_metrics_fn = compute_metrics_t5_multilabel(label_list, tokenizer)
        else:
            label_map = {v: i for i, v in enumerate(label_list)}
            compute_metrics_fn = compute_metrics_t5(tokenizer, label_map)
    else:
        if is_multilabel:
            compute_metrics_fn = compute_metrics_multilabel(label_list)
        else:
            compute_metrics_fn = compute_metrics

    output_name = output_file.replace(".txt", "")
    train_dir = f"temp/train_{output_name}"
    if model_key == "t5":
        training_args = Seq2SeqTrainingArguments(
            output_dir=train_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=100,
            weight_decay=0.01,
            logging_steps=10,
            logging_dir="temp/logs",
            disable_tqdm=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=1,
            predict_with_generate=True,
        )
    else:
        training_args = TrainingArguments(
        output_dir=train_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        # gradient_accumulation_steps=4,
        num_train_epochs=100,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir="temp/logs",
        disable_tqdm=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if model_key == "t5":
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            compute_metrics=compute_metrics_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            data_collator=data_collator,
        )

    trainer.train()

    preds = trainer.predict(tokenized_test)

    if model_key == "t5":
        if is_multilabel:
            decoded_preds = tokenizer.batch_decode(preds.predictions, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(preds.label_ids, skip_special_tokens=True)

            decoded_preds = [p.strip().lower().split(",") for p in decoded_preds]
            decoded_labels = [l.strip().lower().split(",") for l in decoded_labels]

            mlb = MultiLabelBinarizer(classes=label_list)
            mlb.fit([label_list])

            y_pred = mlb.transform(decoded_preds)
            y_true = mlb.transform(decoded_labels)

            report = classification_report(
                y_true,
                y_pred,
                target_names=label_list,
                digits=4,
                zero_division=0
            )
        else:
            report = t5_classification_report(preds, tokenizer, test_df, label_list)
    else:
        if is_multilabel:
            probs = torch.sigmoid(torch.tensor(preds.predictions)).numpy()
            y_pred = (probs >= 0.5).astype(int)
            y_true = preds.label_ids

            report = classification_report(
                y_true,
                y_pred,
                labels=list(range(len(label_list))),
                target_names=label_list,
                digits=4,
                zero_division=0
            )
        else:
            report = classification_report(
                preds.label_ids,
                preds.predictions.argmax(-1),
                labels=list(range(len(label_list))),
                target_names=[str(l) for l in label_list],
                digits=4,
            )

    with open(output_file, 'a') as f:
        f.write("\n=== Experiment Result ({}) ===\n".format(datetime.now()))
        f.write(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True)
    parser.add_argument("-da", required=True)
    parser.add_argument("-model", required=True)
    parser.add_argument("-iteration", type=int, default=5)
    args = parser.parse_args()

    output_file = f"output_{args.dataset}_{args.da}_{args.model}.txt"
    is_multilabel = args.dataset.startswith("tweets")

    for i in range(args.iteration):
        train_df, val_df, test_df = load_and_split_dataset(args.dataset)
        start_time = time.time()
        train_df = apply_da_script(args.dataset, args.da, train_df)
        train_model(args.model, train_df, val_df, test_df, output_file, is_multilabel)
        elapsed_time = time.time() - start_time
        with open(output_file, 'a') as f:
            f.write(f"\n[Iteration {i + 1}] Time elapsed: {elapsed_time:.2f} seconds\n")


if __name__ == "__main__":
    main()
