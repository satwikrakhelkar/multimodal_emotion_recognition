import os
import csv

# Emotion mapping
EMOTION_MAP = {
    "angry": "angry",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "surprise": "surprise"
}

# Paths
DATA_PATH = "data/TESS/"
OUTPUT_CSV = "data/fusion_metadata.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "transcript", "label"])

    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)
        if not os.path.isdir(folder_path):
            continue

        # Extract emotion from folder name
        emotion = folder.split("_")[-1].lower()
        if emotion not in EMOTION_MAP:
            continue

        for file in os.listdir(folder_path):
            if file.endswith(".wav"):
                filepath = os.path.join(folder_path, file)

                # Transcript placeholder (replace with actual transcript if available)
                transcript = file.replace(".wav", "")

                writer.writerow([filepath, transcript, emotion])

print(f"Fusion metadata CSV created at {OUTPUT_CSV}")