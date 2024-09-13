import torch
from torch.utils.data import Dataset

class TestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.sentences = df.iloc[:, 1:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1).values
        self.labels = df.iloc[:, 0].values - 1  # Label을 0부터 시작하도록 조정
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        inputs = self.tokenizer(sentence, truncation=True, max_length=512, padding='max_length', return_tensors="pt")
        label = torch.tensor(self.labels[idx])
        return inputs, label