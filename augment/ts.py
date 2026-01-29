import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForMaskedLM


def is_valid_text(text):
    return isinstance(text, str) and text.strip().lower() not in ["", "nan", "none"]


def text_smoothing_augment(text, model, tokenizer, embedding_layer, smooth_rate=0.5, temp=1.5, device="cpu"):
    if not is_valid_text(text):
        print(f"[WARNING] Skipping invalid text: {repr(text)}")
        return text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids", torch.zeros_like(input_ids)).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        logits = outputs.logits

    probs = F.softmax(logits / temp, dim=-1)
    one_hot = torch.zeros_like(probs).scatter_(2, input_ids.unsqueeze(-1), 1.0)
    smoothed_probs = smooth_rate * one_hot + (1 - smooth_rate) * probs

    token_ids = torch.argmax(smoothed_probs, dim=-1).squeeze().tolist()
    aug_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return aug_text


def main():
    parser = argparse.ArgumentParser(description="Text Smoothing")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with 'text', 'label'")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file to save augmented data")
    parser.add_argument("-naug", type=int, default=8, help="Number of augmentations per sample")
    parser.add_argument("--smooth_rate", type=float, default=0.5, help="Interpolation strength")
    parser.add_argument("--temp_rate", type=float, default=1.0, help="Temperature for softmax")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns")

    # 필터링: 유효한 텍스트만 남기기
    df = df[df["text"].apply(is_valid_text)]

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    model.eval()

    embedding_layer = model.bert.embeddings.word_embeddings
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model.to(device)

    augmented_data = []
    for idx, row in df.iterrows():
        orig_text = row["text"]
        label = row["label"]
        augmented_data.append({"text": orig_text, "label": label})
        for _ in range(args.naug):
            aug_text = text_smoothing_augment(
                orig_text, model, tokenizer, embedding_layer,
                smooth_rate=args.smooth_rate, temp=args.temp_rate, device=device
            )
            augmented_data.append({"text": aug_text, "label": label})

    pd.DataFrame(augmented_data).to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
