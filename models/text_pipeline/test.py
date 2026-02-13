import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load dataset
df = pd.read_csv("data/text_data.csv", on_bad_lines="skip").dropna()
texts = df["text"].tolist()
labels = df["label"].astype(int).tolist()

# Load tokenizer and model from saved directory
tokenizer = BertTokenizer.from_pretrained("models/text_pipeline/text_model")
model = BertForSequenceClassification.from_pretrained("models/text_pipeline/text_model")

# Dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Prepare data
dataset = TextDataset(texts, labels, tokenizer)
loader = DataLoader(dataset, batch_size=8, shuffle=False)

# Test loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Text Test Accuracy: {100 * correct / total:.2f}%")