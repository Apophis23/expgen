import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_path, device):
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model.to(device)
    return model, tokenizer

def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device