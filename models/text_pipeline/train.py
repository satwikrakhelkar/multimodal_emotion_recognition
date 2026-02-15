from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from dataset_loader import TextDataset

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=7)

# Load training dataset
train_dataset = TextDataset(
    csv_file="data/text_data.csv",
    split="train",
    tokenizer_name="bert-base-uncased"
)

# Training configuration
training_args = TrainingArguments(
    output_dir="./results_text",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs_text",
    logging_steps=50
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)

# Train and save model
trainer.train()
model.save_pretrained("models/text_pipeline/text_model")
tokenizer.save_pretrained("models/text_pipeline/text_model")
