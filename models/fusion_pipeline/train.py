import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import librosa
import numpy as np
from transformers import BertTokenizer, BertModel

# Emotion mapping (same as speech pipeline)
EMOTION_MAP = {
    "angry": 0, "disgust": 1, "fear": 2,
    "happy": 3, "neutral": 4, "sad": 5, "surprise": 6
}

# Fusion dataset: loads audio + transcript
class FusionDataset(Dataset):
    def __init__(self, metadata_csv, sr=16000, n_mels=64, max_len=200):
        self.data = []
        self.labels = []
        self.sr, self.n_mels, self.max_len = sr, n_mels, max_len

        df = pd.read_csv(metadata_csv)  # columns: filepath, transcript, label
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        for _, row in df.iterrows():
            filepath, transcript, emotion = row["filepath"], row["transcript"], row["label"]
            if emotion not in EMOTION_MAP:
                continue
            label = EMOTION_MAP[emotion]

            # Speech features
            y, sr = librosa.load(filepath, sr=self.sr)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - np.mean(mel_db)) / (np.std(mel_db) + 1e-6)
            if mel_db.shape[1] < self.max_len:
                pad_width = self.max_len - mel_db.shape[1]
                mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
            else:
                mel_db = mel_db[:, :self.max_len]
            speech_tensor = torch.tensor(mel_db.T, dtype=torch.float32)

            # Text features
            encoding = self.tokenizer(transcript, truncation=True, padding="max_length",
                                      max_length=64, return_tensors="pt")
            text_ids = encoding["input_ids"].squeeze(0)
            text_mask = encoding["attention_mask"].squeeze(0)

            self.data.append((speech_tensor, text_ids, text_mask))
            self.labels.append(label)

        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        speech, ids, mask = self.data[idx]
        return speech, ids, mask, self.labels[idx]

# Fusion model: CNN+LSTM for speech + BERT for text
class FusionModel(nn.Module):
    def __init__(self, num_classes=7):
        super(FusionModel, self).__init__()
        # Speech branch
        self.conv1 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=128, hidden_size=128, num_layers=2,
                            batch_first=True, dropout=0.3)

        # Text branch (BERT)
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # Fusion + classifier
        self.fc1 = nn.Linear(128 + 768, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, speech, ids, mask):
        # Speech branch
        x = speech.transpose(1, 2)  # (batch, n_mels, time)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        speech_vec = out[:, -1, :]  # last time step

        # Text branch
        text_out = self.bert(input_ids=ids, attention_mask=mask)
        text_vec = text_out.pooler_output

        # Fusion
        fused = torch.cat((speech_vec, text_vec), dim=1)
        fused = self.fc1(fused)
        fused = self.relu(fused)
        fused = self.dropout(fused)
        return self.fc2(fused)

def train():
    dataset = FusionDataset("data/fusion_metadata.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(2):
        model.train()
        total_loss = 0
        for speech, ids, mask, labels in train_loader:
            speech, ids, mask, labels = speech.to(device), ids.to(device), mask.to(device), labels.to(device)
            outputs = model(speech, ids, mask)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), "Results/fusion_model.pth")
    print("Fusion model saved.")

if __name__ == "__main__":
    train()