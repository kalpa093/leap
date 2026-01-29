import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import csv
import os
from transformers import AutoTokenizer, AutoModelForMaskedLM
from utils import hf_masked_encode, hf_reconstruction_prob_tok, fill_batch

def gen_neighborhood(args, lines, labels, output_path_txt):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    r_model = AutoModelForMaskedLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    r_model.eval()
    if torch.cuda.is_available():
        r_model.cuda()

    softmax_mask = np.full(len(tokenizer.vocab), False)
    softmax_mask[tokenizer.all_special_ids] = True
    for k, v in tokenizer.vocab.items():
        if '[unused' in k:
            softmax_mask[v] = True

    lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]
    assert len(lines) <= 2, "Only single sentences or sentence pairs can be encoded."

    contexts = None
    if args.context:
        contexts = [tuple(s.strip().split('\t')) for s in open(args.context, encoding='utf-8').readlines()]
        contexts = [[[s] for s in s_list] for s_list in list(zip(*contexts))]
        assert not (len(lines) == 1 and len(contexts) == 2), "Too many contexts for inputs"
        if len(lines) == 2 and len(contexts) == 1:
            contexts = contexts.append(contexts[0])

    sents = []
    l = []
    context_list = [] if contexts is not None else None
    num_gen = []
    gen_index = []
    num_tries = []
    next_sent = 0

    s_rec_file = open(output_path_txt, 'w', encoding='utf-8')
    l_rec_file = open(output_path_txt.replace('.txt', '.label'), 'w', encoding='utf-8')

    sents, l, context_list, next_sent, num_gen, num_tries, gen_index = fill_batch(
        args, tokenizer, sents, l, context_list, lines, contexts, labels,
        next_sent, num_gen, num_tries, gen_index
    )

    while (sents != []):
        for i in range(len(num_gen))[::-1]:
            if num_gen[i] == args.num_samples or num_tries[i] > args.max_tries:
                gen_sents = sents.pop(i)
                num_gen.pop(i)
                gen_index.pop(i)
                label = l.pop(i)

                for sg in gen_sents[1:]:
                    text = ' '.join([repr(val)[1:-1] for val in sg])
                    s_rec_file.write(text + '\n')
                    l_rec_file.write(label + '\n')

        sents, l, context_list, next_sent, num_gen, num_tries, gen_index = fill_batch(
            args, tokenizer, sents, l, context_list, lines, contexts, labels,
            next_sent, num_gen, num_tries, gen_index
        )

        if len(sents) == 0:
            break

        toks = []
        masks = []
        for i in range(len(gen_index)):
            s = sents[i][gen_index[i]]
            c = context_list[i] if context_list is not None else None
            tok, mask = hf_masked_encode(
                tokenizer,
                s,
                context=c,
                noise_prob=args.noise_prob,
                random_token_prob=args.random_token_prob,
                leave_unmasked_prob=args.leave_unmasked_prob,
            )
            tok = tok[:args.max_len]
            mask = mask[:args.max_len]
            toks.append(tok)
            masks.append(mask)

        max_len = min(args.max_len, max([len(tok) for tok in toks]))
        pad_tok = tokenizer.pad_token_id
        toks = [F.pad(tok, (0, max_len - len(tok)), 'constant', pad_tok) for tok in toks]
        masks = [F.pad(mask, (0, max_len - len(mask)), 'constant', pad_tok) for mask in masks]
        toks = torch.stack(toks)
        masks = torch.stack(masks)

        if torch.cuda.is_available():
            toks = toks.cuda()
            masks = masks.cuda()

        rec, rec_masks = hf_reconstruction_prob_tok(toks, masks, tokenizer, r_model, softmax_mask, reconstruct=True, topk=args.topk)

        for i in range(len(rec)):
            rec_work = rec[i].cpu().tolist()
            s_rec = [s.strip() for s in tokenizer.decode([val for val in rec_work if val != tokenizer.pad_token_id][1:-1]).split(tokenizer.sep_token)]
            s_rec = tuple(s_rec)
            if s_rec not in sents[i] and '' not in s_rec:
                sents[i].append(s_rec)
                num_gen[i] += 1
                num_tries[i] = 0
                gen_index[i] = 0
            else:
                num_tries[i] += 1
                gen_index[i] += 1
                if gen_index[i] == len(sents[i]):
                    gen_index[i] = 0

        del toks
        del masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with text and label columns.')
    parser.add_argument('--output', type=str, required=True, help='Output path prefix (without extension).')
    parser.add_argument('--context', type=str, default=None)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--tokenizer', type=str, default=None)
    parser.add_argument('--noise-prob', type=float, default=0.15)
    parser.add_argument('--random-token-prob', type=float, default=0.1)
    parser.add_argument('--leave-unmasked-prob', type=float, default=0.1)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--max-tries', type=int, default=10)
    parser.add_argument('--min-len', type=int, default=4)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--topk', type=int, default=-1)

    args = parser.parse_args()

    if not args.tokenizer:
        args.tokenizer = args.model

    df = pd.read_csv(args.input)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    # Clean tab characters
    df["text"] = df["text"].astype(str).str.replace("\t", " ", regex=False)
    df["text"] = df["text"].astype(str).str.replace(",", " ", regex=False)

    lines = [(x,) for x in df["text"].tolist()]
    labels = df["label"].astype(str).tolist()

    txt_path = args.output + ".txt"
    gen_neighborhood(args, lines, labels, txt_path)

    # Read generated augmented data
    with open(txt_path, encoding="utf-8") as f_text, open(txt_path.replace(".txt", ".label"), encoding="utf-8") as f_label:
        df_aug = pd.DataFrame({
            "text": [line.strip() for line in f_text],
            "label": [line.strip() for line in f_label]
        })

    # Merge original and augmented
    df_orig = df[["text", "label"]].copy()
    df_final = pd.concat([df_orig, df_aug], ignore_index=True)
    df_final = df_final[df_final.notnull().all(axis=1)]  # Drop rows with NaNs
    df_final = df_final[df_final.apply(
        lambda row: isinstance(row["text"], str) and isinstance(row["label"], str), axis=1)]
    df_final = df_final[
        df_final["text"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0) &
        df_final["label"].apply(lambda x: isinstance(x, str) and len(x.strip()) > 0)
        ]

    # Save final output
    df_final.to_csv(args.output, index=False, quoting=csv.QUOTE_ALL)
