import torch
from torch.utils.data import DataLoader
from train import FusionDataset, FusionModel  # reuse dataset + model

def test():
    dataset = FusionDataset("data/fusion_metadata.csv")
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    model = FusionModel()
    model.load_state_dict(torch.load("Results/fusion_model.pth"))
    model.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            # Adjust unpacking based on FusionDataset output
            speech_features = batch[0]
            text_features = batch[1]
            mask = batch[2]          # <-- new
            labels = batch[3]        # <-- new

            outputs = model(speech_features, text_features, mask)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Fusion Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    test()