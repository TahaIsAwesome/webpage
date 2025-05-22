import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

# 1. Load and prepare data
df = pd.read_csv("dataset.csv")
# Convert labels to numeric
label_map = {"human": 0, "ai": 1}
df["label"] = df["label"].map(label_map)

# Split data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df)
})

# 2. Tokenize
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

dataset = dataset.map(tokenize_function, batched=True)

# 3. Model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 4. Training
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer
)

# 5. Train
trainer.train()

# 6. Save model
trainer.save_model("ai-human-detector")
tokenizer.save_pretrained("ai-human-detector")
