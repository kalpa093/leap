from transformers import AutoModelForSequenceClassification
from transformers.models.deberta_v2 import DebertaV2TokenizerFast

def load_model(num_labels):
    model_name = "microsoft/deberta-v3-large"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model

def load_model_multi(num_labels):
    model_name = "microsoft/deberta-v3-large"
    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return tokenizer, model
