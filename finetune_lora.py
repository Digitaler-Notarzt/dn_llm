import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)

# Configuration
model_name = "akjindal53244/Llama-3.1-Storm-8B"
max_seq_length = 2048
output_dir = "./lora_finetuned_llama_storm"

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
    max_memory={i: "8GiB" for i in range(torch.cuda.device_count())}
)
model = prepare_model_for_kbit_training(model)  # Critical fix

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token

# Load and preprocess dataset
dataset = load_dataset("json", data_files="trainData.json", split="train")

def preprocess(examples):
    instructions = [f"### Instruction:\n{ex}\n" for ex in examples['input']]
    responses = [f"### Response:\n{ex}</s>" for ex in examples['response']]
    sequences = [inst + resp for inst, resp in zip(instructions, responses)]

    tokenized = tokenizer(
        sequences,
        max_length=max_seq_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

# LoRA configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    inference_mode=False
)

# Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    bf16=torch.cuda.is_bf16_supported(),
    gradient_checkpointing=True,
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch",
    report_to="none",
    remove_unused_columns=False,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine"
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

# Train and save
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete! Model saved to", output_dir)

