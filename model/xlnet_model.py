from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model(num_labels):
    model_name = "xlnet-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    return tokenizer, model
def load_model_multi(num_labels):
    model_name = "xlnet-large-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    return tokenizer, model
