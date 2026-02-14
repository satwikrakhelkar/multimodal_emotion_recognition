import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_loader import SpeechDataset
from model_speech import SpeechModel

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001

# Dataset
train_data = SpeechDataset(split="train")
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = SpeechModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training loop
for epoch in range(EPOCHS):
    for inputs, labels in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "speech_model.pth")
