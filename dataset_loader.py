import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd

# -----------------------------
# Speech Dataset
# -----------------------------
class SpeechDataset(Dataset):
    def __init__(self, csv_file="data/speech_metadata.csv", split="train", transform=None):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split]
        self.transform = transform

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["emotion"].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filepath = self.data.iloc[idx]["file_path"]
        emotion = self.data.iloc[idx]["emotion"]

        waveform, sr = torchaudio.load(filepath)
        mel_spec = torchaudio.transforms.MelSpectrogram(sr)(waveform)

        if self.transform:
            mel_spec = self.transform(mel_spec)

        label = self.label_map[emotion]
        return mel_spec, torch.tensor(label)


# -----------------------------
# Text Dataset
# -----------------------------
class TextDataset(Dataset):
    def __init__(self, csv_file="data/text_data.csv", split="train", tokenizer_name="bert-base-uncased", max_len=128):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split]
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["emotion"].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]["transcript"])
        emotion = self.data.iloc[idx]["emotion"]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        label = self.label_map[emotion]
        return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), torch.tensor(label)


# -----------------------------
# Fusion Dataset
# -----------------------------
class FusionDataset(Dataset):
    def __init__(self, csv_file="data/fusion_metadata.csv", split="train", tokenizer_name="bert-base-uncased", max_len=128):
        self.data = pd.read_csv(csv_file)
        self.data = self.data[self.data["split"] == split]
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

        self.label_map = {label: idx for idx, label in enumerate(sorted(self.data["emotion"].unique()))}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Speech
        speech_path = self.data.iloc[idx]["speech_file"]
        waveform, sr = torchaudio.load(speech_path)
        mel_spec = torchaudio.transforms.MelSpectrogram(sr)(waveform)

        # Text
        text = str(self.data.iloc[idx]["transcript"])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Label
        emotion = self.data.iloc[idx]["emotion"]
        label = self.label_map[emotion]

        return mel_spec, encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0), torch.tensor(label)
