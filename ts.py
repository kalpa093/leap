import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM, AutoTokenizer


def is_valid_text(text):
    return isinstance(text, str) and text.strip().lower() not in {"", "nan", "none"}


def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_token_ids(probs, input_ids, attention_mask, tokenizer, decode_strategy, top_k):
    """Decode paper-style soft token distributions into text for CSV-based training."""
    sampled_ids = input_ids.clone()
    special_ids = set(tokenizer.all_special_ids)
    seq_len = input_ids.size(1)

    for pos in range(seq_len):
        token_id = int(input_ids[0, pos].item())
        if not bool(attention_mask[0, pos].item()) or token_id in special_ids:
            continue

        dist = probs[0, pos]
        if top_k and top_k > 0:
            k = min(top_k, dist.numel())
            top_probs, top_ids = torch.topk(dist, k=k)
            top_probs = top_probs / top_probs.sum().clamp_min(1e-12)
            if decode_strategy == "argmax":
                sampled_ids[0, pos] = top_ids[torch.argmax(top_probs)]
            else:
                sampled_ids[0, pos] = top_ids[torch.multinomial(top_probs, 1).item()]
        elif decode_strategy == "argmax":
            sampled_ids[0, pos] = torch.argmax(dist)
        else:
            sampled_ids[0, pos] = torch.multinomial(dist, 1).item()

    return sampled_ids


def text_smoothing_augment(
    text,
    model,
    tokenizer,
    smooth_rate=0.1,
    temp=1.0,
    max_length=256,
    decode_strategy="sample",
    top_k=50,
    device="cpu",
):
    if not is_valid_text(text):
        return text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    token_type_ids = inputs.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    # The paper enables MLM dropout instead of explicitly masking tokens.
    model.train()
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        logits = outputs.logits

    probs = F.softmax(logits / temp, dim=-1)
    one_hot = F.one_hot(input_ids, num_classes=probs.size(-1)).float()
    smoothed_probs = smooth_rate * one_hot + (1.0 - smooth_rate) * probs
    smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    token_ids = sample_token_ids(
        smoothed_probs,
        input_ids,
        attention_mask,
        tokenizer,
        decode_strategy=decode_strategy,
        top_k=top_k,
    )
    aug_text = tokenizer.decode(token_ids.squeeze(0).tolist(), skip_special_tokens=True)
    return aug_text if is_valid_text(aug_text) else text


def main():
    parser = argparse.ArgumentParser(description="Text Smoothing for CSV-based experiments")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file with text,label columns")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file")
    parser.add_argument("-naug", type=int, default=8, help="Number of decoded augmentations per sample")
    parser.add_argument(
        "--smooth_rate",
        type=float,
        default=0.1,
        help="Paper lambda: weight assigned to the original one-hot token distribution",
    )
    parser.add_argument(
        "--temp_rate",
        type=float,
        default=1.0,
        help="Softmax temperature. 1.0 matches the paper's ordinary softmax.",
    )
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--decode_strategy", choices=["sample", "argmax"], default="sample")
    parser.add_argument("--top_k", type=int, default=50, help="Limit sampled MLM candidates; use 0 for full vocab")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain text and label columns")

    df = df[df["text"].apply(is_valid_text)].copy()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).to(device)

    augmented_data = []
    for _, row in df.iterrows():
        orig_text = row["text"]
        label = row["label"]
        augmented_data.append({"text": orig_text, "label": label})

        for _ in range(args.naug):
            aug_text = text_smoothing_augment(
                orig_text,
                model,
                tokenizer,
                smooth_rate=args.smooth_rate,
                temp=args.temp_rate,
                max_length=args.max_length,
                decode_strategy=args.decode_strategy,
                top_k=args.top_k,
                device=device,
            )
            augmented_data.append({"text": aug_text, "label": label})

    pd.DataFrame(augmented_data).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
