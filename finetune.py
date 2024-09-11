from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import torch

#raw_datasets = load_dataset("Demonthos/tailwind-components")

device = torch.device("mps")

raw_datasets = load_dataset("yelp_review_full")

tokenizer = AutoTokenizer.from_pretrained("akjindal53244/Llama-3.1-Storm-8B")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained("akjindal53244/Llama-3.1-Storm-8B", num_labels=5).to(device)


training_args = TrainingArguments(output_dir="test_trainer")

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
