import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from train import FusionModel, FusionDataset

def extract_embeddings(mode="fusion"):
    dataset = FusionDataset("data/fusion_metadata.csv")
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionModel().to(device)
    model.load_state_dict(torch.load("Results/fusion_model.pth"))
    model.eval()

    embeddings, labels = [], []
    with torch.no_grad():
        for speech, ids, mask, lbls in loader:
            speech, ids, mask = speech.to(device), ids.to(device), mask.to(device)

            # Speech branch
            x = speech.transpose(1, 2)
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = x.transpose(1, 2)
            out, _ = model.lstm(x)
            speech_vec = out[:, -1, :]

            # Text branch
            text_out = model.bert(input_ids=ids, attention_mask=mask)
            text_vec = text_out.pooler_output

            # Choose mode
            if mode == "speech":
                vec = speech_vec
            elif mode == "text":
                vec = text_vec
            else:  # fusion
                vec = torch.cat((speech_vec, text_vec), dim=1)

            embeddings.append(vec.cpu())
            labels.extend(lbls.tolist())

    embeddings = torch.cat(embeddings, dim=0).numpy()
    return embeddings, labels

def plot_embeddings(embeddings, labels, method="tsne", mode="fusion"):
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = reducer.fit_transform(embeddings)
        fname = f"Results/{mode}_tsne.png"
        title = f"t-SNE {mode.capitalize()} Embeddings"
    else:
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        fname = f"Results/{mode}_pca.png"
        title = f"PCA {mode.capitalize()} Embeddings"

    plt.figure(figsize=(8,6))
    scatter = plt.scatter(reduced[:,0], reduced[:,1], c=labels, cmap="tab10", alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Emotions")
    plt.title(title)
    plt.savefig(fname)
    plt.close()  # prevents PyCharm freeze

if __name__ == "__main__":
    for mode in ["speech", "text", "fusion"]:
        embeddings, labels = extract_embeddings(mode=mode)
        plot_embeddings(embeddings, labels, method="tsne", mode=mode)
        plot_embeddings(embeddings, labels, method="pca", mode=mode)
    print("All plots saved in Results/")