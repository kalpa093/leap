import os
import time
import argparse
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from datasets import Dataset
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig
from sklearn.preprocessing import LabelEncoder
from subprocess import run

QLORA_MODEL_MAP = {
    "llama3": "meta-llama/Meta-Llama-3-8B",
    "codegen": "Salesforce/codegen2-7b",
    "phi": "microsoft/phi-2",
    "deepseek": "deepseek-ai/deepseek-llm-7b-base"
}

torch.cuda.empty_cache()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["MKL_THREADING_LAYER"] = "INTEL"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


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
    os.makedirs("temp", exist_ok=True)
    train_df.to_csv(input_path, index=False)
    run(["python", f"augment/{da_method}.py", "--input", input_path, "--output", output_path])
    return pd.read_csv(output_path)


def load_qlora_model(model_key):
    model_id = QLORA_MODEL_MAP[model_key]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    return tokenizer, model


def build_train_prompt(text, label_text):
    return f'Classify the sentiment of this text: {text}\nAnswer: {label_text}'


def build_eval_prompt(text, label_list, is_multilabel=False):
    if is_multilabel:
        guide = f"Choose one or more labels (comma-separated) from: {', '.join(label_list)}."
    else:
        guide = f"Choose exactly one label from: {', '.join(label_list)}."
    return f'Classify the sentiment of this text.\n{guide}\nText: "{text}"\nAnswer:'


def tokenize_train(example, tokenizer, is_multilabel=False):
    if is_multilabel:
        label_text = ", ".join(example["label"]) if isinstance(example["label"], list) else example["label"]
    else:
        label_text = example["label"]

    pre = f'Classify the sentiment of this text: {example["text"]}\nAnswer: '
    full = pre + label_text

    tok_pre  = tokenizer(pre,  truncation=True, padding="max_length", max_length=256)
    tok_full = tokenizer(full, truncation=True, padding="max_length", max_length=256)

    input_ids = tok_full["input_ids"]
    attn      = tok_full["attention_mask"]

    seq_len_full = int(sum(attn))
    seq_len_pre  = int(sum(tok_pre["attention_mask"]))
    max_len      = len(attn)

    start_idx    = max_len - seq_len_full
    answer_start = start_idx + seq_len_pre
    answer_end   = start_idx + seq_len_full

    labels = [-100] * max_len
    for i in range(answer_start, answer_end):
        labels[i] = input_ids[i]

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def tokenize_eval(example, tokenizer, label_list, is_multilabel=False):
    prompt = build_eval_prompt(example["text"], label_list, is_multilabel)
    tok = tokenizer(prompt, truncation=True, padding="max_length", max_length=256)
    return {"input_ids": tok["input_ids"], "attention_mask": tok["attention_mask"]}


def predict_on_dataset_with_generate(model, tokenizer, dataset, batch_size=4, max_new_tokens=10):
    model.eval()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            decoded_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_preds.extend(decoded_texts)
    return all_preds


def extract_label_from_output_single(text):
    try:
        answer = text.split("Answer:")[-1].strip()
        first = answer.split("\n")[0].split(",")[0].split()[0].lower()
        return first.strip(".,:;!?|<>\"'()[]")
    except Exception:
        return "none"


def extract_labels_from_output_multi(text):
    try:
        answer = text.split("Answer:")[-1].strip()
        line = answer.split("\n")[0]
        parts = [p.strip().lower().strip(".,:;!?|<>\"'()[]") for p in line.split(",")]
        return [p for p in parts if p]
    except Exception:
        return []


def safe_binarize_and_classify(decoded_preds, decoded_labels, label_list):
    if all(isinstance(lbl, str) for lbl in decoded_preds):
        decoded_preds = [[lbl] for lbl in decoded_preds]
    if all(isinstance(lbl, str) for lbl in decoded_labels):
        decoded_labels = [[lbl] for lbl in decoded_labels]

    filtered_preds = [[label for label in pred if label in label_list] for pred in decoded_preds]
    filtered_labels = [[label for label in label if label in label_list] for label in decoded_labels]

    filtered_preds = [pred if pred else ["none"] for pred in filtered_preds]
    filtered_labels = [label if label else ["none"] for label in filtered_labels]

    extended_label_list = label_list + (["none"] if "none" not in label_list else [])
    mlb = MultiLabelBinarizer(classes=extended_label_list)
    y_pred = mlb.fit_transform(filtered_preds)
    y_true = mlb.transform(filtered_labels)

    assert y_true.shape == y_pred.shape, "Shape of y_true and y_pred."

    report = classification_report(
        y_true,
        y_pred,
        target_names=extended_label_list,
        digits=4,
        zero_division=0
    )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", required=True)
    parser.add_argument("-model", required=True, choices=QLORA_MODEL_MAP.keys())
    parser.add_argument("-da", required=True)
    parser.add_argument("-iteration", type=int, default=3)
    args = parser.parse_args()

    output_file = f"output_qlora_{args.dataset}_{args.da}_{args.model}.txt"
    is_multilabel = args.dataset.startswith("tweets")

    for i in range(args.iteration):
        train_df, val_df, test_df = load_and_split_dataset(args.dataset)
        train_df = apply_da_script(args.dataset, args.da, train_df)

        for df in [train_df, val_df, test_df]:
            df["text"] = df["text"].astype(str)
            df["label"] = df["label"].astype(str).str.lower()
            if is_multilabel:
                df["label"] = df["label"].apply(lambda x: x.split(",") if "," in x else [x])

        label_list = sorted(set(l for d in train_df["label"] for l in (d if isinstance(d, list) else [d])))
        label_encoder = LabelEncoder()
        label_encoder.fit(label_list)

        tokenizer, model = load_qlora_model(args.model)

        train_dataset = Dataset.from_pandas(train_df).map(
            lambda x: tokenize_train(x, tokenizer, is_multilabel)
        )
        val_dataset = Dataset.from_pandas(val_df).map(
            lambda x: tokenize_train(x, tokenizer, is_multilabel)
        )

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir = f"temp/train_{args.dataset}_{args.da}_{args.model}_{i}",
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                num_train_epochs=2,
                learning_rate=2e-5,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                logging_dir="temp/logs",
                save_total_limit=1,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                overwrite_output_dir=True,
            ),
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        )

        start = time.time()
        trainer.train()

        test_df = test_df.dropna(subset=["text"]).reset_index(drop=True)
        test_dataset = Dataset.from_pandas(test_df).map(
            lambda x: tokenize_eval(x, tokenizer, label_list, is_multilabel)
        )
        test_dataset = test_dataset.remove_columns(
            [c for c in test_dataset.column_names if c not in ["input_ids", "attention_mask"]]
        )
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        raw_preds = predict_on_dataset_with_generate(model, tokenizer, test_dataset, batch_size=4, max_new_tokens=10)
        elapsed = time.time() - start

        if is_multilabel:
            y_true = test_df["label"].tolist()
        else:
            y_true = [d if isinstance(d, str) else d[0] for d in test_df["label"].tolist()]

        if is_multilabel:
            pred_lists = [extract_labels_from_output_multi(t) for t in raw_preds]
            pred_lists = [[l for l in pl if l in label_list] or ["none"] for pl in pred_lists]
            report = safe_binarize_and_classify(pred_lists, y_true, label_list)
        else:
            pred_single = [extract_label_from_output_single(t) for t in raw_preds]
            pred_single = [p if p in label_list else "none" for p in pred_single]
            extended = label_list + (["none"] if "none" not in label_list else [])
            report = classification_report(
                y_true,
                pred_single,
                labels=extended,
                target_names=extended,
                digits=4,
                zero_division=0
            )

        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== Experiment ({datetime.now()}) ===\n")
            f.write(report)
            f.write(f"\nTime elapsed: {elapsed:.2f} seconds\n")


if __name__ == "__main__":
    main()
