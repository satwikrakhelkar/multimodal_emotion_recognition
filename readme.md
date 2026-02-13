# Multimodal Emotion Recognition

## 📌 Overview
This project implements a **multimodal emotion recognition pipeline** that fuses speech, text, and visual modalities to classify human emotions.  
It combines deep learning models for each modality and integrates them into a fusion model for improved accuracy and robustness.

## 🚀 Features
- **Speech pipeline**: Extracts acoustic features and classifies emotional tone.
- **Text pipeline**: Processes transcripts using transformer-based models.
- **Fusion model**: Combines outputs from speech and text for final emotion prediction.
- **Visualization tools**: Embedding analysis with PCA/t-SNE for interpretability.
- **Reproducible setup**: Requirements tracked in `requirements.txt`, large models managed via Git LFS.

## 📂 Repository Structure

multimodal_emotion_recognition/
├── data/                     # Raw datasets (external, not versioned)
│   ├── speech/               # Speech/audio data
│   ├── text/                 # Text transcripts
│   └── fusion/               # Preprocessed multimodal data
│
├── src/                      # Core source code
│   ├── preprocessing/        # Data cleaning & feature extraction
│   ├── models/               # Model architectures
│   │   ├── speech_pipeline/  # Speech emotion model
│   │   ├── text_pipeline/    # Text emotion model
│   │   └── fusion_model/     # Fusion logic
│   ├── training/             # Training scripts
│   ├── evaluation/           # Evaluation scripts & metrics
│   └── visualization/        # PCA/t-SNE plots, embedding analysis
│
├── Results/                  # Deliverables
│   ├── metrics/              # Accuracy tables, confusion matrices
│   ├── plots/                # Graphs, PCA/t-SNE visualizations
│   └── reports/              # Final evaluation reports
│
├── configs/                  # Experiment configs (YAML/JSON)
│
├── models/                   # Saved weights (Git LFS tracked)
│   ├── speech_model.pth
│   ├── text_model.safetensors
│   └── fusion_model.pth
│
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Ignore rules for clean repo
## ⚙️ Installation
Clone the repository and set up the environment:
```bash
git clone https://github.com/satwikrakhelkar/multimodal_emotion_recognition.git
cd multimodal_emotion_recognition
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows
pip install -r requirements.txt


📊 Usage
Train and evaluate the fusion model:
python src/train_fusion.py --config configs/fusion.yaml


Run evaluation and generate results:
python src/evaluate.py --model Results/fusion_model.pth


📈 Results
- Accuracy tables and error analysis are available in Results/.
- Embedding visualizations (PCA/t-SNE) provide insights into modality fusion.
🛠️ Tech Stack
- Python 3.9+
- PyTorch for deep learning
- Transformers for text modeling
- Git LFS for large model files
📌 Notes
- Large files (*.pth, *.safetensors) are tracked via Git LFS.
- Datasets are not included due to size; please add them manually in data/.
👨‍💻 Author
Developed by Satwik Rakhelkar
Final-year Electronics & Communication Engineering student, Matrusri Engineering College.
Internship experience at ISRO and Vishwam.AI, with expertise in AI/ML, robotics, and embedded systems.



