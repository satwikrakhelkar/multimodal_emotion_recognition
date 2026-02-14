                                                                                                                         
Multimodal Emotion Recognition
## 📌 Objective
This project implements emotion recognition using three approaches:
• 	Speech-only pipeline
• 	Text-only pipeline
• 	Multimodal fusion pipeline (speech + text)
Dataset: Toronto Emotional Speech Set (TESS)

## 🚀 Features
• 	Speech pipeline: Extracts acoustic features (MFCCs, spectrograms) and classifies emotional tone.
• 	Text pipeline: Processes transcripts using transformer-based models for contextual meaning.
• 	Fusion model: Combines outputs from speech and text for final emotion prediction.
• 	Visualization tools: Embedding analysis with PCA/t-SNE for interpretability.
• 	Reproducible setup: Dependencies tracked in , large models managed via Git LFS.

## 📂 Repo Structure
• 	project/
• 	models/
• 	 → Training & testing scripts for speech-only model
• 	 → Training & testing scripts for text-only model
• 	 → Training & testing scripts for multimodal fusion model
• 	preprocessing/ → Scripts for speech (silence trimming, resampling) and text (tokenization, cleaning)
• 	feature_extraction/ → Scripts for speech features (MFCCs, spectrograms) and text features (embeddings)
• 	utils/ → Dataset loader, visualization, metrics
• 	Results/ → Accuracy tables, error analysis, visualizations
• 	Report/ → Final report with architectures, experiments, analysis
• 	 → Dependencies
• 	 → Setup instructions, usage, repo overview

## ⚙️ Installation
Clone the repository and set up the environment:
git clone https://github.com/satwikrakhelkar/multimodal_emotion_recognition.git
cd multimodal_emotion_recognition

python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate      # On Windows

pip install -r requirements.txt

## 📂 Datasets
This project uses the **Toronto Emotional Speech Set (TESS)** dataset, available on Kaggle:

- https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess

The dataset contains speech samples along with corresponding transcripts and emotion labels.

## 🛠 Setup
1. Download the dataset from Kaggle.
2. Place it in the `data/`
3. Run preprocessing scripts:
   
run:
python src/preprocessing/preprocess_speech.py
python src/preprocessing/preprocess_text.pyRun preprocessing scripts:

run:
python src/preprocessing/preprocess_speech.py
python src/preprocessing/preprocess_text.py


## 📊 Usage
Speech‑only pipeline:
python src/models/speech_pipeline/train.py
python src/models/speech_pipeline/test.py

Text‑only pipeline:
python src/models/text_pipeline/train.py
python src/models/text_pipeline/test.py

Fusion pipeline:
python src/models/fusion_pipeline/train.py --config configs/fusion.yaml
python src/models/fusion_pipeline/test.py --model Results/fusion_model.pth


## 📈 Results
Performance on held‑out test sets:

Model Variant        Accuracy    Notes
------------------------------------------------------------
Speech-only           15.38%      Poor convergence, weak classification
Text-only             28.57%      Undertrained, limited contextual learning
Fusion (Speech+Text)  100.00%     Perfect separation, strong multimodal benefit

- Accuracy tables and error analysis are available in Results/.
- Confusion matrices and metrics are in metrics/.
- PCA/t‑SNE plots are in plots/.

## 🛠️ Tech Stack
• 	Python 3.9+
• 	PyTorch for deep learning
• 	HuggingFace Transformers for text modeling
• 	Git LFS for large model files

## 📌 Notes
• 	Large files (, ) are tracked via Git LFS.
Evaluators must install Git LFS before cloning to access full model files:
git lfs install
git clone https://github.com/satwikrakhelkar/multimodal_emotion_recognition.git
git lfs pull

• 	Datasets are not included due to size; please add them manually in .

## 👨‍💻 Author
Satwik Rakhelkar
Final‑year Electronics & Communication Engineering student, Matrusri Engineering College.
Internship experience at ISRO and Vishwam.AI, with expertise in AI/ML, robotics, and embedded systems.








