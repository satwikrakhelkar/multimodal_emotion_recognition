Multimodal Emotion Recognition
📌 Overview
This project implements a multimodal emotion recognition pipeline that fuses speech, text, and visual modalities to classify human emotions.
By combining deep learning models for each modality and integrating them into a fusion model, the system achieves improved accuracy and robustness.

🚀 Features
• 	Speech pipeline: Extracts acoustic features and classifies emotional tone.
• 	Text pipeline: Processes transcripts using transformer‑based models.
• 	Fusion model: Combines outputs from speech and text for final emotion prediction.
• 	Visualization tools: Embedding analysis with PCA/t‑SNE for interpretability.
• 	Reproducible setup: Dependencies tracked in , large models managed via Git LFS.

📂 Repository Structure
multimodal_emotion_recognition/
├── src/
├── models/
├── Results/
├── configs/
├── metrics/
├── plots/
├── reports/
├── requirements.txt
├── README.md
└── .gitignore

⚙️ Installation
Clone the repository and set up the environment:
git clone https://github.com/satwikrakhelkar/multimodal_emotion_recognition.git
cd multimodal_emotion_recognition

python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r requirements.txt

## 📂 Datasets
This project uses the **Toronto Emotional Speech Set (TESS)** dataset, available on Kaggle:

- [Toronto Emotional Speech Set (TESS)] (https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

The dataset contains speech samples along with corresponding transcripts and emotion labels.

### Setup
1. Download the dataset from Kaggle.
2. Place it in the `data/`
3. Run preprocessing scripts:
```bash
python src/preprocessing/preprocess_speech.py
python src/preprocessing/preprocess_text.pyRun preprocessing scripts:
```bash
python src/preprocessing/preprocess_speech.py
python src/preprocessing/preprocess_text.py


📊 Usage
Train and evaluate the fusion model:
python src/train_fusion.py --config configs/fusion.yaml

Run evaluation and generate results:
python src/evaluate.py --model Results/fusion_model.pth


📈 Results
• 	Accuracy tables and error analysis are available in .
• 	Embedding visualizations (PCA/t‑SNE) provide insights into modality fusion.

🛠️ Tech Stack
• 	Python 3.9+
• 	PyTorch for deep learning
• 	HuggingFace Transformers for text modeling
• 	Git LFS for large model files

📌 Notes
• 	Large files (, ) are tracked via Git LFS.
Evaluators must install Git LFS before cloning to access full model files:
git lfs install
git clone https://github.com/satwikrakhelkar/multimodal_emotion_recognition.git
git lfs pull

• 	Datasets are not included due to size; please add them manually in .

👨‍💻 Author
Satwik Rakhelkar
Final‑year Electronics & Communication Engineering student, Matrusri Engineering College.
Internship experience at ISRO and Vishwam.AI, with expertise in AI/ML, robotics, and embedded systems.

