import torch
from torch.utils.data import DataLoader
from dataset_loader import SpeechDataset
from model_speech import SpeechModel

test_data = SpeechDataset(split="test")
test_loader = DataLoader(test_data, batch_size=32)

model = SpeechModel()
model.load_state_dict(torch.load("speech_model.pth"))
model.eval()

correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Speech pipeline accuracy: {100 * correct / total:.2f}%")
