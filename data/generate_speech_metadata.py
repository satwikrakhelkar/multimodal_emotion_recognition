import os
import csv
import random

# Root directory where TESS dataset is stored
DATASET_DIR = "data/TESS"
OUTPUT_FILE = "data/speech_metadata.csv"

# Emotions expected in TESS dataset
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Train/test split ratio
TRAIN_RATIO = 0.8

def generate_metadata():
    rows = []
    for emotion in EMOTIONS:
        emotion_dir = os.path.join(DATASET_DIR, emotion)
        if not os.path.exists(emotion_dir):
            print(f"Warning: {emotion_dir} not found, skipping {emotion}")
            continue

        files = [f for f in os.listdir(emotion_dir) if f.endswith(".wav")]
        random.shuffle(files)

        split_index = int(len(files) * TRAIN_RATIO)
        train_files = files[:split_index]
        test_files = files[split_index:]

        for f in train_files:
            rows.append([os.path.join(emotion_dir, f), emotion, "train"])
        for f in test_files:
            rows.append([os.path.join(emotion_dir, f), emotion, "test"])

    # Write to CSV
    with open(OUTPUT_FILE, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["file_path", "emotion", "split"])
        writer.writerows(rows)

    print(f"Metadata saved to {OUTPUT_FILE}, total {len(rows)} entries.")

if __name__ == "__main__":
    generate_metadata()
