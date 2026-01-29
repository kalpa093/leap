import sys
import os
import torch
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import csv
import argparse
import re
from subprocess import run
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments, 
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training, 
    TaskType,
    PeftModel
)
from datasets import Dataset
import torch.nn.functional as F

print(f"[DEBUG] Raw Arguments received: {sys.argv}")

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", "--dataset", type=str, required=True, help="Dataset name")
parser.add_argument("-da", "--da", type=str, required=True, help="Data Augmentation method")
parser.add_argument("-model", "--model", type=str, default="xlnet", help="Model type")
parser.add_argument("-iteration", "--iteration", type=int, default=1, help="Iteration number")
parser.add_argument("-samples", "--samples", type=int, default=1, help="Number of samples for SHAP analysis")

args, unknown = parser.parse_known_args()

DATASET_NAME = args.dataset
DA_METHOD = args.da
NUM_SAMPLES = args.samples
MODEL_TYPE = args.model

if MODEL_TYPE == "phi2":
    MODEL_NAME = "microsoft/phi-2"
    USE_QLORA = True
elif MODEL_TYPE == "codebert":
    MODEL_NAME = "microsoft/codebert-base"
    USE_QLORA = False
elif MODEL_TYPE == "xlnet":
    MODEL_NAME = "xlnet-base-cased"
    USE_QLORA = False
else:
    MODEL_NAME = "xlnet-base-cased"
    USE_QLORA = False

print(f"[DEBUG] Configuration -> Dataset: {DATASET_NAME}, DA: {DA_METHOD}, Model: {MODEL_NAME}, QLoRA: {USE_QLORA}")

TRAIN_DATA_PATH = f"dataset/{DATASET_NAME}.csv"
SAMPLE_DATA_PATH = "dataset/github_sample.csv"

if USE_QLORA:
    OUTPUT_DIR = f"temp_shap_{MODEL_TYPE}_qlora"
else:
    OUTPUT_DIR = f"temp_shap_{MODEL_TYPE}_full"

MAX_LEN = 256
BATCH_SIZE = 8 if USE_QLORA else 16
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABEL_LIST = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for i, label in enumerate(LABEL_LIST)}

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
env = os.environ.copy()

def get_code_mask(text, tokenizer, max_len=None):
    code_ranges = []
    for m in re.finditer(r"```[\s\S]*?```", text):
        code_ranges.append(m.span())
    for m in re.finditer(r"`[^`\n]+`", text):
        code_ranges.append(m.span())

    try:
        encoding = tokenizer(
            text, 
            return_offsets_mapping=True, 
            add_special_tokens=True, 
            truncation=True, 
            max_length=max_len if max_len else MAX_LEN
        )
        offsets = encoding.offset_mapping 
    except NotImplementedError:
        print("[WARN] Offset mapping not supported. Skipping code highlighting details.")
        tokens = tokenizer.tokenize(text)
        return tokens, np.zeros((1, len(tokens)))

    tokens = tokenizer.convert_ids_to_tokens(encoding.input_ids)
    
    is_code_mask = []
    for (token_start, token_end) in offsets:
        if token_start == token_end: 
            is_code_mask.append(0)
            continue
        is_code = 0
        for (code_start, code_end) in code_ranges:
            if max(token_start, code_start) < min(token_end, code_end):
                is_code = 1
                break
        is_code_mask.append(is_code)
    
    return tokens, np.array([is_code_mask])

def load_and_split_dataset():
    print(f"[INFO] Loading FULL Training Dataset from: {TRAIN_DATA_PATH}...")
    if os.path.exists(TRAIN_DATA_PATH):
        try:
            df = pd.read_csv(TRAIN_DATA_PATH, encoding='windows-1252', quotechar='"')
        except:
            df = pd.read_csv(TRAIN_DATA_PATH, encoding='utf-8', quotechar='"')
    else:
        raise FileNotFoundError(f"[ERROR] Training data not found at: {TRAIN_DATA_PATH}")

    df = df.dropna(subset=['text', 'label'])
    if DATASET_NAME == "github":
        valid_labels = ["positive", "neutral", "negative"]
        df = df[df['label'].astype(str).str.lower().isin(valid_labels)]
    df['label'] = df['label'].astype(str).str.lower()

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    return train_df, val_df, test_df

def apply_da_script(dataset_name, da_method, train_df):
    if da_method == "c2l":
        c2l_path = f"temp/{dataset_name}_c2l.csv"
        if os.path.exists(c2l_path):
            return pd.read_csv(c2l_path)
        return train_df
    if da_method == "none":
        return train_df

    print(f"[DA] Applying external augmentation script: {da_method}...")
    if not os.path.exists("temp"): os.makedirs("temp")
    
    input_path = f"temp/{dataset_name}_orig.csv"
    output_path = f"temp/{dataset_name}_{da_method}.csv"
    
    quoting_opt = csv.QUOTE_ALL if dataset_name == "github" else csv.QUOTE_MINIMAL
    train_df.to_csv(input_path, index=False, quoting=quoting_opt)

    script_path = f"augment/{da_method}.py"
    if not os.path.exists(script_path): return train_df

    try:
        run(["python", script_path, "--input", input_path, "--output", output_path], env=env, check=True)
    except Exception as e:
        print(f"[ERROR] DA Script Failed: {e}")
        return train_df

    if os.path.exists(output_path):
        return pd.read_csv(output_path)
    return train_df

def tokenize_function(example, tokenizer):
    text = str(example["text"]) if example["text"] is not None else ""
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LEN)
    tokenized["labels"] = LABEL2ID.get(str(example["label"]), 1) 
    return tokenized

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = predictions.argmax(-1)
    acc = accuracy_score(labels, decoded_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, decoded_preds, average='weighted', zero_division=0)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

def train_and_save():
    print(f"[START] Pipeline: Dataset={DATASET_NAME}, DA={DA_METHOD}, Model={MODEL_NAME}")
    torch.cuda.empty_cache()

    train_df, val_df, test_df = load_and_split_dataset()
    train_df = apply_da_script(DATASET_NAME, DA_METHOD, train_df)
    train_df = train_df.dropna(subset=['text', 'label'])
    train_df['text'] = train_df['text'].astype(str)
    train_df['label'] = train_df['label'].astype(str)
    
    print(f"[INFO] Loading Tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_QLORA:
        print(f"[INFO] QLoRA Enabled: Loading {MODEL_NAME} in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(LABEL_LIST),
            quantization_config=bnb_config, 
            trust_remote_code=True,
            device_map="auto" # QLoRA일 때만 사용
        )
        model = prepare_model_for_kbit_training(model)
        
        peft_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["Wqkv", "out_proj", "fc1", "fc2"], 
            lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, peft_config)
        
    else:
        print(f"[INFO] Full Fine-tuning Enabled: Loading {MODEL_NAME} (FP32/FP16)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, 
            num_labels=len(LABEL_LIST),
            trust_remote_code=True
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    
    tokenized_train = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=False, remove_columns=train_ds.column_names)
    tokenized_val = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=False, remove_columns=val_ds.column_names)
    
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-4 if USE_QLORA else 2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,
        disable_tqdm=False,
        fp16=True if torch.cuda.is_available() else False,
        optim="paged_adamw_32bit" if USE_QLORA else "adamw_torch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    
    print("[INFO] Starting Training...")
    trainer.train()
    
    print(f"[SAVE] Saving model to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    return test_df

def run_shap_visualization():
    print(f"\n[INFO] Starting SHAP Analysis on SAMPLE DATA ({SAMPLE_DATA_PATH})...")
    torch.cuda.empty_cache()

    try:
        tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, use_fast=True, trust_remote_code=True)
    except:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if USE_QLORA:
        print("[INFO] Loading Base Model + Adapter (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(LABEL_LIST), quantization_config=bnb_config,
            trust_remote_code=True, device_map="auto"
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    else:
        print("[INFO] Loading Fine-tuned Model (Full)...")
        model = AutoModelForSequenceClassification.from_pretrained(
            OUTPUT_DIR, trust_remote_code=True
        ).to(DEVICE)
    
    model.eval()

    def f_predictor(texts):
        text_list = texts.tolist() if isinstance(texts, pd.Series) else texts
        text_list = [str(t) for t in text_list]

        inputs = tokenizer(
            text_list, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=MAX_LEN
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        probs = F.softmax(outputs.logits.float(), dim=-1).cpu().numpy()
        return probs

    masker = shap.maskers.Text(tokenizer)
    explainer = shap.Explainer(f_predictor, masker, output_names=LABEL_LIST)

    if os.path.exists(SAMPLE_DATA_PATH):
        try:
             sample_df = pd.read_csv(SAMPLE_DATA_PATH)
        except:
             sample_df = pd.read_csv(SAMPLE_DATA_PATH, encoding='windows-1252', quotechar='"')
        if 'label' in sample_df.columns:
             sample_df['label'] = sample_df['label'].astype(str).str.lower()
        sample_texts = sample_df['text'].iloc[:NUM_SAMPLES].tolist()
        true_labels = sample_df['label'].iloc[:NUM_SAMPLES].tolist()
    else:
        sample_texts = ["test code"]
        true_labels = ["neutral"]

    for idx, (text, true_label) in enumerate(zip(sample_texts, true_labels)):
        print(f"   Generating Heatmap for Sample {idx+1}/{len(sample_texts)}...")
        if len(str(text)) > 2000: text = str(text)[:2000]

        shap_values = explainer([str(text)])
        values = shap_values.values[0] 
        tokens, token_types_arr = get_code_mask(str(text), tokenizer, max_len=None)
        
        shap_len = values.shape[0]
        mask_len = token_types_arr.shape[1]
        if mask_len > shap_len:
            token_types_arr = token_types_arr[:, :shap_len]
        elif mask_len < shap_len:
            pad_width = shap_len - mask_len
            token_types_arr = np.pad(token_types_arr, ((0,0), (0, pad_width)), 'constant')

        heatmap_data = values.T 
        fig = plt.figure(figsize=(15, 6))
        gs = fig.add_gridspec(2, 2,  width_ratios=[50, 1], height_ratios=[10, 1], wspace=0.02, hspace=0.05)

        ax1 = fig.add_subplot(gs[0, 0]) 
        ax2 = fig.add_subplot(gs[1, 0]) 
        cbar_ax = fig.add_subplot(gs[0, 1]) 

        sns.heatmap(
            heatmap_data, ax=ax1, cbar_ax=cbar_ax, xticklabels=False, yticklabels=LABEL_LIST,
            cmap="coolwarm_r", center=0, annot=False, cbar_kws={'label': 'SHAP Contribution'}
        )
        ax1.set_title(f"Sample {idx+1} ({DA_METHOD.upper()}) [{MODEL_NAME}] SHAP\nTrue: {true_label}")
        ax1.set_ylabel("Class")

        cmap_type = mcolors.ListedColormap(['#e0e0e0', '#0044ff']) 
        sns.heatmap(
            token_types_arr, ax=ax2, cbar=False, xticklabels=50, yticklabels=["Type"],
            cmap=cmap_type, vmin=0, vmax=1
        )
        ax2.set_xlabel("Token Index (Blue=Code, Gray=NLP)")
        ax2.set_yticks([]) 
        
        ax1.set_xlim(0, shap_len)
        ax2.set_xlim(0, shap_len)

        save_path = f"{OUTPUT_DIR}/sample_{idx+1}_{DA_METHOD}_shap.pdf"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"   [DONE] Saved: {save_path}")

if __name__ == "__main__":
    train_and_save()
    run_shap_visualization()
