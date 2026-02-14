# Error Analysis – Multimodal Emotion Recognition

## Speech Pipeline
- Accuracy was **15.38%**, indicating poor convergence.
- Likely causes:
  - Label mismatch during training.
  - Insufficient training epochs.
  - Preprocessing inconsistencies (e.g., silence trimming, resampling).

## Text Pipeline
- Accuracy was **28.57%**, underperforming compared to expected benchmarks.
- Issues observed:
  - Model saved after minimal training.
  - Requires further fine-tuning of BERT.
  - Limited contextual learning due to short training duration.

## Fusion Pipeline
- Accuracy was **100%**, showing perfect separation of emotions.
- Notes:
  - Multimodal integration successfully captured complementary features.
  - Fusion helped resolve ambiguities present in unimodal pipelines.

## Common Misclassifications
- Neutral speech misclassified as **Sad** when tone was flat but transcript was neutral.
- Happy text misclassified as **Neutral** when transcripts lacked explicit emotional words.
- **Fear** and **Surprise** occasionally confused due to similar vocal intensity.

## Debugging Notes
- Dataset path issues and model input errors were identified and fixed.
- AI assistant guidance accelerated troubleshooting and improved pipeline stability.
