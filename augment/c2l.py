import os

os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
import csv
import argparse
import seaborn as sn
import matplotlib.pyplot as plt
import pickle
from transformers import BertForMaskedLM

parser = argparse.ArgumentParser(description="Triplet Generation Script")
parser.add_argument('--input', type=str, required=True, help="Input CSV file path")
parser.add_argument('--output', type=str, required=True, help="Output CSV file path")
args = parser.parse_args()

INPUT_CSV_PATH = args.input
OUTPUT_CSV_PATH = args.output

SPLIT_SAMPLES = 1000000
NOTEBOOK_INDEX = 0
PICKLE_PATH = f"{os.path.splitext(INPUT_CSV_PATH)[0]}_cf_augmented_examples"

os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
        self.length = len(labels)

    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if isinstance(val, torch.Tensor):
                item[key] = val[idx]
            else:
                item[key] = torch.tensor(val[idx])
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return self.length


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


data_df = pd.read_csv(INPUT_CSV_PATH, encoding='windows-1252')
data_df['text'] = data_df['text'].apply(lambda x: x if isinstance(x, str) else "" if pd.isnull(x) else str(x))

IS_MULTILABEL = False
if os.path.basename(INPUT_CSV_PATH).lower().startswith("tweets") or \
   data_df['label'].apply(lambda x: isinstance(x, list)).any() or \
   data_df['label'].apply(lambda x: isinstance(x, str) and ',' in x).any():

    IS_MULTILABEL = True
    from sklearn.preprocessing import MultiLabelBinarizer
    data_df['label'] = data_df['label'].apply(lambda x: x if isinstance(x, list) else [lbl.strip() for lbl in str(x).split(',')])
    mlb = MultiLabelBinarizer()
    binarized_labels = mlb.fit_transform(data_df['label'])
    NUM_LABELS = len(mlb.classes_)
    label_mapping = {label: idx for idx, label in enumerate(mlb.classes_)}
    reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    train_labels = binarized_labels[SPLIT_SAMPLES * NOTEBOOK_INDEX: SPLIT_SAMPLES * (NOTEBOOK_INDEX + 1)]
else:
    data_df['label'] = data_df['label'].astype(str)
    unique_labels = sorted(data_df['label'].unique())
    label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
    reverse_label_mapping = {idx: label for label, idx in label_mapping.items()}
    NUM_LABELS = len(unique_labels)
    train_labels = [label_mapping[lbl] for lbl in data_df['label'].tolist()[SPLIT_SAMPLES * NOTEBOOK_INDEX: SPLIT_SAMPLES * (NOTEBOOK_INDEX + 1)]]

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=NUM_LABELS,
    problem_type='multi_label_classification' if IS_MULTILABEL else None
).to(device)

train_texts = data_df['text'].tolist()[SPLIT_SAMPLES * NOTEBOOK_INDEX: SPLIT_SAMPLES * (NOTEBOOK_INDEX + 1)]
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_dataset = IMDbDataset(train_encodings, train_labels)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)


def get_gradient_norms(batch):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    if IS_MULTILABEL:
        labels = labels.float()
    else:
        labels = labels.long()

    # For CrossEntropy Loss
    #_, labels = torch.max(labels, dim=1)
    # labels = batch['labels'].item()

    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward(retain_graph=True)
    torch.cuda.empty_cache()

    importances = torch.tensor([]).to(device)
    for pos_index, token_index in zip(range(1, len(input_ids[0])), input_ids[0][1:]):
        if token_index == tokenizer.sep_token_id:
            break

        importance = torch.norm(model.bert.embeddings.position_embeddings.weight.grad[pos_index], 2).float().detach()
        importances = torch.cat((importances, importance.unsqueeze(0)), dim=-1)

    model.bert.embeddings.position_embeddings.weight.grad = None

    # return importances_list
    return importances

# Compute gradient at BERT's position_embeddings (discard [cls] and [sep]/[pad])
# Only works for batch_size = 1
def get_gradient_norms(batch):
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    #_, labels = torch.max(labels, dim=1)
    if IS_MULTILABEL:
        labels = labels.float()
    else:
        labels = labels.long()
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    loss.backward(retain_graph=True)
    torch.cuda.empty_cache()

    importances = torch.tensor([]).to(device)
    for pos_index, token_index in zip(range(1, len(input_ids[0])), input_ids[0][1:]):
        if token_index == tokenizer.sep_token_id:
            break
        importance = torch.norm(model.bert.embeddings.position_embeddings.weight.grad[pos_index], 2).float().detach()
        importances = torch.cat((importances, importance.unsqueeze(0)), dim=-1)

    model.bert.embeddings.position_embeddings.weight.grad = None
    return importances

"""## Utility Functions"""

def visualize(words, masks):
    fig, ax = plt.subplots(figsize=(len(words), 1))
    plt.rc('xtick', labelsize=16)
    heatmap = sn.heatmap([masks], xticklabels=words, yticklabels=False, square=True, \
                         linewidths=0.1, cmap='coolwarm', center=0.5, vmin=0, vmax=1)
    plt.xticks(rotation=45)
    plt.show()


def mask_causal_words(tokens, importances, topk=1):
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1][:topk]
    for topk_idx in topk_indices:
#         print(topk_idx)
#         print(tokens[topk_idx])
        causal_mask[topk_idx] = 1

    return causal_mask

def compute_importances(data_loader, importance_function):
    all_importances = []
    for batch in tqdm(data_loader):
        importances = importance_function(batch)
        all_importances.append(importances)
    return all_importances


def compute_average_importance(data_loader, all_importances):
    all_averaged_importances = []
    importance_dict = dict()
    importance_dict_counter = dict()

    for importances, batch in tqdm(zip(all_importances, data_loader)):
        tokens = [x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]]

        for tok_imp, tok in zip(importances, tokens):
            if not tok in importance_dict.keys():
                importance_dict[tok.item()] = 0
                importance_dict_counter[tok.item()] = 0
            importance_dict[tok.item()] += tok_imp
            importance_dict_counter[tok.item()] += 1


    for importances, batch in tqdm(zip(all_importances, data_loader)):
        tokens = [x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]]
        averaged_importances = torch.tensor([importance_dict[x.item()]/importance_dict_counter[x.item()] for x in tokens])
        all_averaged_importances.append(averaged_importances)

    return all_averaged_importances

"""## Gradient-based Causal Masking Function"""

def build_causal_mask_with_precomputed(data_loader, all_importances, sampling_ratio, augment_ratio, label_map):
    triplets = []
    error_cnt = 0
    for importances, batch in tqdm(zip(all_importances, data_loader)):
        tokens = torch.tensor([x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]])
        assert tokens.size() == importances.size()

        orig_sample = tokenizer.decode(tokens)
        causal_mask = mask_causal_words(tokens.cpu().numpy(), importances.cpu().numpy(), topk=sampling_ratio)
        # visualize(tokens, causal_mask)
        # print(causal_mask)

        if 1 not in causal_mask:
            # print(orig_sample[1], cf_sample[1])
            continue

        for _ in range(augment_ratio):
            # 모든 causal 단어를 mask, 모든 non-causal 단어를 mask
            if sampling_ratio is None:
                causal_masked_tokens = [tokens[i] if causal_mask[i] == 0 else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if causal_mask[i] == 1 else tokenizer.mask_token_id for i in range(len(tokens))]

            # sampling_ratio 갯수 (int) 만큼의 단어를 mask
            elif type(sampling_ratio) == int:
                causal_indices = np.where(np.array(causal_mask) == 1)[0]
                noncausal_indices = np.where(np.array(causal_mask) == 0)[0]

                # print(causal_indices)

                causal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                try:
                    noncausal_mask_indices = np.random.choice(noncausal_indices, max(1, min(sampling_ratio, len(noncausal_indices))))
                    #noncausal_mask_indices = np.random.choice(noncausal_indices, 1)
                except:
                    noncausal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                    error_cnt += 1

                causal_masked_tokens = [tokens[i] if i not in causal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if i not in noncausal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]

            # sampling_ratio 비율 (%) 만큼의 단어를 mask
            else:
                pass

            causal_masked_sample = tokenizer.decode(causal_masked_tokens)
            noncausal_masked_sample = tokenizer.decode(noncausal_masked_tokens)

            #_, labels = torch.max(batch['labels'], dim=1)
            labels = batch['labels'].item()
            label_idx = labels[0].item()
            label = label_map.get(label_idx, f"Label_{label_idx}")
            triplets.append((label, orig_sample, causal_masked_sample, noncausal_masked_sample, False, 0))
    print(f"Error Cnt: {error_cnt}")
    return triplets, 0



"""## Propensity-based Causal Masking Function"""

def _TVD(orig_logits, cf_logits):
    return 0.5 * torch.cdist(orig_logits.unsqueeze(0), cf_logits.unsqueeze(0), p=1).squeeze().item()
"""For Debugging"""
"""
flipped_TVD = []
unflipped_TVD = []
def mask_softed_propensity_causal_words(tokens, batch, importances, topk=1):
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    orig_outputs = model(input_ids, attention_mask=attention_mask)
    _, orig_prediction = torch.max(orig_outputs[0], dim=1)
    for i, topk_idx in enumerate(topk_indices):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0][topk_idx + 1] = tokenizer.mask_token_id
        masked_outputs = model(masked_input_ids, attention_mask=attention_mask)
        _, masked_prediction = torch.max(masked_outputs[0], dim=1)
        if orig_prediction != masked_prediction:
            #DEBUGGING
            flipped_TVD.append(_TVD(torch.softmax(orig_outputs[0], dim=-1), torch.softmax(masked_outputs[0], dim=-1)))
            #DEBUGGING END
            causal_mask[topk_idx] = 1
            break
        else:
            #DEBUGGING
            unflipped_TVD.append(_TVD(torch.softmax(orig_outputs[0], dim=-1), torch.softmax(masked_outputs[0], dim=-1)))
            #DEBUGGING END

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True
    return causal_mask, err_flag
"""
def mask_uniform_propensity_causal_words(tokens, batch, importances, topk=1):
    THRESHOLD = 0.1
    uniform_dist = torch.ones(1, 2) / 2.0
    uniform_dist = uniform_dist.to("cuda")
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    orig_outputs = model(input_ids, attention_mask=attention_mask)
    _, orig_prediction = torch.max(orig_outputs[0], dim=1)
    best_tvd = 0.
    best_idx = -1
    for i, topk_idx in enumerate(topk_indices):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0][topk_idx + 1] = tokenizer.mask_token_id
        masked_outputs = model(masked_input_ids, attention_mask=attention_mask)
        _, masked_prediction = torch.max(masked_outputs[0], dim=1)
        orig_tvd = _TVD(torch.softmax(orig_outputs[0], dim=-1), uniform_dist)
        masked_tvd = _TVD(torch.softmax(masked_outputs[0], dim=-1), uniform_dist)

        # Use maximum value
        if orig_tvd > masked_tvd and abs(orig_tvd - masked_tvd) > best_tvd:
            causal_mask[best_idx] = 0
            causal_mask[topk_idx] = 1
            best_tvd = abs(orig_tvd - masked_tvd)
            best_idx = topk_idx
        else:
            continue
        """
        # Use Gradient Order
        if orig_tvd > masked_tvd:
            causal_mask[topk_idx] = 1
            break
        else:
            continue
        """

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True
    return causal_mask, err_flag, best_tvd

def mask_softed_propensity_causal_words(tokens, batch, importances, topk=1):
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    orig_outputs = model(input_ids, attention_mask=attention_mask)
    _, orig_prediction = torch.max(orig_outputs[0], dim=1)
    best_tvd = 0.
    best_idx = -1
    for i, topk_idx in enumerate(topk_indices):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0][topk_idx + 1] = tokenizer.mask_token_id
        masked_outputs = model(masked_input_ids, attention_mask=attention_mask)
        _, masked_prediction = torch.max(masked_outputs[0], dim=1)
        tvd_value = _TVD(torch.softmax(orig_outputs[0], dim=-1), torch.softmax(masked_outputs[0], dim=-1))

        # Use Maximum  Value
        if tvd_value > best_tvd:
            causal_mask[best_idx] = 0
            causal_mask[topk_idx] = 1
            best_tvd = tvd_value
            best_idx = topk_idx
        else:
            continue

        """
        # Use Gradeint Order
        if orig_tvd > MIN_FLIPPED:
            causal_mask[topk_idx] = 1
            break
        else:
            continue
        """

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True
    return causal_mask, err_flag, best_tvd

def mask_propensity_causal_words(tokens, batch, importances, topk=1):
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    topk_indices = topk_indices[:len(tokens)]
    err_flag = False
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    orig_outputs = model(input_ids, attention_mask=attention_mask)
    _, orig_prediction = torch.max(orig_outputs[0], dim=1)
    for i, topk_idx in enumerate(topk_indices):
        masked_input_ids = input_ids.clone()
        masked_input_ids[0][topk_idx + 1] = tokenizer.mask_token_id
        masked_outputs = model(masked_input_ids, attention_mask=attention_mask)
        _, masked_prediction = torch.max(masked_outputs[0], dim=1)
        if orig_prediction != masked_prediction:
            causal_mask[topk_idx] = 1
            break
    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True
    return causal_mask, err_flag, 0

def build_propensity_causal_mask_with_precomputed(data_loader, all_importances, sampling_ratio, augment_ratio, label_map):
    triplets = []
    error_cnt = 0
    no_flip_cnt = 0
    no_flip_idx = []
    for importances, batch in tqdm(zip(all_importances, data_loader)):
        tokens = torch.tensor([x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]])
        assert tokens.size() == importances.size()
        if len(tokens) == 0:
            print(batch['input_ids'][0])
            triplets.append((label, orig_sample, orig_sample, orig_sample, True, 0))
            continue
        orig_sample = tokenizer.decode(tokens)
        #causal_mask, err_flag, maximum_score = mask_propensity_causal_words(tokens.cpu().numpy(), batch, importances.cpu().numpy(), topk=sampling_ratio)
        #causal_mask, err_flag, maximum_score = mask_softed_propensity_causal_words(tokens.cpu().numpy(), batch, importances.cpu().numpy(), topk=sampling_ratio)
        causal_mask, err_flag, maximum_score = mask_uniform_propensity_causal_words(tokens.cpu().numpy(), batch, importances.cpu().numpy(), topk=sampling_ratio)
        no_flip_idx.append(err_flag)
        if err_flag:
            no_flip_cnt += 1
        # visualize(tokens, causal_mask)
        # print(causal_mask)

        if 1 not in causal_mask:
            print(tokens)
            triplets.append((label, orig_sample, orig_sample, orig_sample, err_flag, maximum_score))
            continue

        for _ in range(augment_ratio):
            # 모든 causal 단어를 mask, 모든 non-causal 단어를 mask
            if sampling_ratio is None:
                causal_masked_tokens = [tokens[i] if causal_mask[i] == 0 else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if causal_mask[i] == 1 else tokenizer.mask_token_id for i in range(len(tokens))]

            # sampling_ratio 갯수 (int) 만큼의 단어를 mask
            elif type(sampling_ratio) == int:
                causal_indices = np.where(np.array(causal_mask) == 1)[0]
                noncausal_indices = np.where(np.array(causal_mask) == 0)[0]

                # print(causal_indices)

                causal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                try:
                    noncausal_mask_indices = np.random.choice(noncausal_indices, max(1, min(sampling_ratio, len(noncausal_indices))))
                    #noncausal_mask_indices = np.random.choice(noncausal_indices, 1)
                except:
                    noncausal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                    error_cnt += 1

                causal_masked_tokens = [tokens[i] if i not in causal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if i not in noncausal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]

            # sampling_ratio 비율 (%) 만큼의 단어를 mask
            else:
                pass

            causal_masked_sample = tokenizer.decode(causal_masked_tokens)
            noncausal_masked_sample = tokenizer.decode(noncausal_masked_tokens)

            #_, labels = torch.max(batch['labels'], dim=1)
            labels = batch['labels'].item()
            label_idx = labels[0].item() if isinstance(labels, torch.Tensor) else labels
            label = label_map.get(label_idx, f"Label_{label_idx}")
            triplets.append((label, orig_sample, causal_masked_sample, noncausal_masked_sample, err_flag, maximum_score))
    print(f"Error Cnt: {error_cnt}")
    print(f"No Flip Cnt: {no_flip_cnt}")
    return triplets, no_flip_idx

"""## MLM-based Causal Masking Function"""


mlm_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
mlm_model = mlm_model.to(device)
mlm_model.eval()
TOPK_NUM = 4

def mask_efficient_LM_dropout_causal_words(tokens, batch, importances, topk=1):
    if len(tokens) == 0:
        return [0], True, 0

    dropout = torch.nn.Dropout(0.5)
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False
    find_flag = False

    input_ids = batch['input_ids']
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # [1, seq_len]
    input_ids = input_ids.expand(TOPK_NUM, -1).to(device)

    attention_mask = batch['attention_mask']
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)
    attention_mask = attention_mask.expand(len(tokens), -1).to(device)

    token_type_ids = batch['token_type_ids']
    if token_type_ids.dim() == 1:
        token_type_ids = token_type_ids.unsqueeze(0)
    token_type_ids = token_type_ids.expand(len(tokens), -1).to(device)

    masked_input_ids = batch['input_ids']
    if masked_input_ids.dim() == 1:
        masked_input_ids = masked_input_ids.unsqueeze(0)
    masked_input_ids = masked_input_ids.expand(len(tokens), -1).to(device)

    masked_attention_mask = attention_mask
    masked_token_type_ids = token_type_ids

    fake_labels = torch.ones((len(tokens),))

    masked_train = IMDbDataset({
        'input_ids': masked_input_ids,
        'attention_mask': masked_attention_mask,
        'token_type_ids': masked_token_type_ids,
        'topk_indices': topk_indices
    }, fake_labels)
    masked_train_loader = DataLoader(masked_train, batch_size=4, shuffle=False, drop_last=True)
    logits = []
    for masked_batch in masked_train_loader:
        masked_input_ids = masked_batch['input_ids'].to(device)
        masked_attention_mask = masked_batch['attention_mask'].to(device)
        masked_token_type_ids = masked_batch['token_type_ids'].to(device)
        topk_index = masked_batch['topk_indices'].to(device)
        masked_input_embeds = mlm_model.bert.embeddings.word_embeddings(masked_input_ids)
        for mi_idx, topk_idx in zip(range(masked_input_embeds.size(0)), topk_index):
            masked_input_embeds[mi_idx][topk_idx + 1] = dropout(masked_input_embeds[mi_idx][topk_idx + 1])
        with torch.no_grad():
            outputs = mlm_model(attention_mask=masked_attention_mask, token_type_ids=masked_token_type_ids, inputs_embeds=masked_input_embeds)
            predictions = outputs[0]
            #logits.append(predictions.detach().cpu())

        topk_logits = torch.topk(predictions, TOPK_NUM, dim=-1)[1]
        mask_candidates = [topk_logit[topk_idx + 1] for topk_idx, topk_logit in zip(topk_index, topk_logits)]

        for topk_idx, mask_candidate in zip(topk_index, mask_candidates):
            # For excepting [SEP] token ...
            if importances[topk_idx] == 0:
                continue
            recon_input_ids = input_ids.clone()
            for i, mc in enumerate(mask_candidate):
                recon_input_ids[i][topk_idx + 1] = mc

            if attention_mask.size(0) != recon_input_ids.size(0):
                attention_mask = attention_mask[:recon_input_ids.size(0), :]
            if token_type_ids.size(0) != recon_input_ids.size(0):
                token_type_ids = token_type_ids[:recon_input_ids.size(0), :]

            with torch.no_grad():
                recon_outputs = model(recon_input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                _, recon_prediction = torch.max(recon_outputs[0], dim=1)

            # IF prediction is changed:
            if len(torch.unique(recon_prediction)) != 1:
                causal_mask[topk_idx] = 1
                find_flag = True
                break

        if find_flag:
            break

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True

    return causal_mask, err_flag, 0

def mask_LM_dropout_causal_words(tokens, batch, importances, topk=1):
    dropout = torch.nn.Dropout(0.5)
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False

    if len(tokens) < 1:
        return causal_mask, True, 0

    masked_input_ids = batch['input_ids'].squeeze().repeat((len(tokens), )).reshape(len(tokens), -1).to(device)
    masked_attention_mask = batch['attention_mask'].expand(len(tokens), -1).to(device)
    fake_labels = torch.ones((len(tokens),))
    masked_train = IMDbDataset({'input_ids': masked_input_ids, 'attention_mask': masked_attention_mask, 'topk_indices': topk_indices}, fake_labels)
    masked_train_loader = DataLoader(masked_train, batch_size=8, shuffle=False, drop_last=True)
    logits = []
    for masked_batch in masked_train_loader:
        masked_input_ids = masked_batch['input_ids'].to(device)
        masked_attention_mask = masked_batch['attention_mask'].to(device)
        topk_index = masked_batch['topk_indices'].to(device)
        masked_input_embeds = mlm_model.bert.embeddings.word_embeddings(masked_input_ids)
        for mie, ti in zip(masked_input_embeds, topk_index):
            mie[ti + 1] = dropout(mie[ti + 1])
        with torch.no_grad():
            outputs = mlm_model(attention_mask=masked_attention_mask, inputs_embeds=masked_input_embeds)
            predictions = outputs[0]
            logits.append(predictions)

    logits = torch.cat(logits, dim=0)
    topk_logits = torch.topk(logits, TOPK_NUM, dim=-1)[1]
    mask_candidates = [topk_logit[topk_idx + 1] for topk_idx, topk_logit in zip(topk_indices, topk_logits)]

    input_ids = batch['input_ids'].squeeze().repeat((TOPK_NUM, )).reshape(TOPK_NUM, -1).to(device)
    attention_mask = batch['attention_mask'].expand(TOPK_NUM, -1).to(device)

    for topk_idx, mask_candidate in zip(topk_indices, mask_candidates):
        recon_input_ids = input_ids.clone()
        for i, mc in enumerate(mask_candidate):
            recon_input_ids[i][topk_idx + 1] = mc

        with torch.no_grad():
            recon_outputs = model(recon_input_ids, attention_mask=attention_mask)
            _, recon_prediction = torch.max(recon_outputs[0], dim=1)

        # IF prediction is changed:
        if len(torch.unique(recon_prediction)) != 1:
            causal_mask[topk_idx] = 1
            break

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True

    return causal_mask, err_flag, 0


def mask_LM_causal_words(tokens, batch, importances, topk=1):
    causal_mask = [0 for _ in range(len(tokens))]
    topk_indices = np.argsort(importances)[::-1]
    err_flag = False

    masked_input_ids = batch['input_ids'].squeeze().repeat((len(tokens), )).reshape(len(tokens), -1).to(device)
    masked_attention_mask = batch['attention_mask'].expand(len(tokens), -1).to(device)
    for i, topk_idx in enumerate(topk_indices):
        masked_input_ids[i][topk_idx + 1] = tokenizer.mask_token_id
    fake_labels = torch.ones((len(tokens),))
    masked_train = IMDbDataset({'input_ids': masked_input_ids, 'attention_mask': masked_attention_mask}, fake_labels)
    masked_train_loader = DataLoader(masked_train, batch_size=8, shuffle=False, drop_last=True)
    logits = []
    for masked_batch in masked_train_loader:
        masked_input_ids = masked_batch['input_ids'].to(device)
        masked_attention_mask = masked_batch['attention_mask'].to(device)
        with torch.no_grad():
            outputs = mlm_model(masked_input_ids, attention_mask=masked_attention_mask)
            predictions = outputs[0]
            logits.append(predictions)
    logits = torch.cat(logits, dim=0)
    topk_logits = torch.topk(logits, TOPK_NUM, dim=-1)[1]
    mask_candidates = [topk_logit[topk_idx + 1] for topk_idx, topk_logit in zip(topk_indices, topk_logits)]

    input_ids = batch['input_ids'].squeeze().repeat((TOPK_NUM, )).reshape(TOPK_NUM, -1).to(device)
    attention_mask = batch['attention_mask'].expand(TOPK_NUM, -1).to(device)
    for topk_idx, mask_candidate in zip(topk_indices, mask_candidates):
        recon_input_ids = input_ids.clone()
        for i, mc in enumerate(mask_candidate):
            recon_input_ids[i][topk_idx + 1] = mc

        with torch.no_grad():
            recon_outputs = model(recon_input_ids, attention_mask=attention_mask)
            _, recon_prediction = torch.max(recon_outputs[0], dim=1)

        # IF prediction is changed:
        if len(torch.unique(recon_prediction)) != 1:
            causal_mask[topk_idx] = 1
            break

    if 1 not in causal_mask:
        causal_mask[topk_indices[0]] = 1
        err_flag = True

    return causal_mask, err_flag, 0

def build_LM_causal_mask_with_precomputed(data_loader, all_importances, sampling_ratio, augment_ratio, label_map, is_multilabel):
    triplets = []
    error_cnt = 0
    no_flip_cnt = 0
    no_flip_idx = []

    for importances, batch in tqdm(zip(all_importances, data_loader)):
        tokens = torch.tensor([x for x in batch['input_ids'][0][1:] if x not in [tokenizer.sep_token_id, tokenizer.pad_token_id]])
        if tokens.size() != importances.size():
            print(f"Size mismatch: tokens={tokens.size()}, importances={importances.size()}")
            continue

        orig_sample = tokenizer.decode(tokens)
        causal_mask, err_flag, maximum_score = mask_efficient_LM_dropout_causal_words(
            tokens.cpu().numpy(), batch, importances.cpu().numpy(), topk=sampling_ratio
        )
        no_flip_idx.append(err_flag)
        if err_flag:
            no_flip_cnt += 1

        if 1 not in causal_mask:
            triplets.append(("UNKNOWN", orig_sample, orig_sample, orig_sample, err_flag, maximum_score))
            continue

        for _ in range(augment_ratio):
            if sampling_ratio is None:
                causal_masked_tokens = [tokens[i] if causal_mask[i] == 0 else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if causal_mask[i] == 1 else tokenizer.mask_token_id for i in range(len(tokens))]
            elif isinstance(sampling_ratio, int):
                causal_indices = np.where(np.array(causal_mask) == 1)[0]
                noncausal_indices = np.where(np.array(causal_mask) == 0)[0]

                try:
                    causal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                    noncausal_mask_indices = np.random.choice(noncausal_indices, max(1, min(sampling_ratio, len(noncausal_indices))))
                except:
                    noncausal_mask_indices = np.random.choice(causal_indices, sampling_ratio)
                    error_cnt += 1

                causal_masked_tokens = [tokens[i] if i not in causal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
                noncausal_masked_tokens = [tokens[i] if i not in noncausal_mask_indices else tokenizer.mask_token_id for i in range(len(tokens))]
            else:
                continue

            causal_masked_sample = tokenizer.decode(causal_masked_tokens)
            noncausal_masked_sample = tokenizer.decode(noncausal_masked_tokens)

            raw_label = batch['labels'][0].cpu().numpy()  # batch_size = 1
            if is_multilabel:
                label_indices = [i for i, v in enumerate(raw_label) if v == 1]
                label_names = [label_map[i] for i in label_indices]
                label_str = ",".join(label_names)
            else:
                label_idx = raw_label.item() if isinstance(raw_label, np.ndarray) else raw_label
                label_str = label_map.get(label_idx, f"Label_{label_idx}")

            triplets.append((label_str, orig_sample, causal_masked_sample, noncausal_masked_sample, err_flag, maximum_score))
    print(f"Error Cnt: {error_cnt}")
    print(f"No Flip Cnt: {no_flip_cnt}")
    return triplets, no_flip_idx


"""## Compute or Load Gradient Importance"""

if not os.path.exists(PICKLE_PATH):
    os.makedirs(PICKLE_PATH)

if os.path.exists(os.path.join(PICKLE_PATH, "gradient_importance.pickle")):
    with open(os.path.join(PICKLE_PATH, "gradient_importance.pickle"), 'rb') as f:
        all_importance = pickle.load(f)
else:
    all_importance = []
    for batch in tqdm(train_loader):
        imp = get_gradient_norms(batch)
        all_importance.append(imp)
    with open(os.path.join(PICKLE_PATH, "gradient_importance.pickle"), 'wb') as f:
        pickle.dump(all_importance, f)

"""## Get Average of Gradient Importance"""

averaged_all_importance = compute_average_importance(train_loader, all_importance)
with open(os.path.join(PICKLE_PATH, "gradient_averaged_importance.pickle"), 'wb') as f:
        pickle.dump(averaged_all_importance, f)

averaged_all_importance = averaged_all_importance[SPLIT_SAMPLES * (NOTEBOOK_INDEX):SPLIT_SAMPLES * (NOTEBOOK_INDEX + 1)]

"""## Generate (Original, Causal-masked, Non-causal-masked) Triplets"""
sampling_ratio = 1
augment_ratio = 4
triplets, _ = build_LM_causal_mask_with_precomputed(train_loader, averaged_all_importance, sampling_ratio, augment_ratio, reverse_label_mapping, is_multilabel=IS_MULTILABEL)
df_triplets = pd.DataFrame(triplets, columns=['label', 'original', 'causal_masked', 'noncausal_masked', 'err_flag', 'maximum_score'])
#df_triplets.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
rows = []
for _, row in df_triplets.iterrows():
    label = row['label']

    rows.append({"text": row['original'], "label": label})
    rows.append({"text": row['causal_masked'], "label": label})
    rows.append({"text": row['noncausal_masked'], "label": label})


df_flattened = pd.DataFrame(rows)
df_flattened.to_csv(OUTPUT_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
