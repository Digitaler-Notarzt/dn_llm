import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer

model_name = "akjindal53244/Llama-3.1-Storm-8B"
max_seq_length = 2048

# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 offloading to CPU
)

# Use CPU if no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

max_memory = {
    "cuda": "8GiB",
    "cpu": "100GiB",
}

# Load the model with quantization
model = AutoModelForCausalLM.from_pretrained(
   "akjindal53244/Llama-3.1-Storm-8B",
   max_memory=max_memory
)

tokenizer = AutoTokenizer.from_pretrained("akjindal53244/Llama-3.1-Storm-8B")

# Load dataset
dataset = load_dataset("json", data_files="trainData.json")

# Preprocess the dataset
def preprocess(examples):
    sequences = [f"{examples['input'][i]} {examples['response'][i]}" for i in range(len(examples['input']))]
    
    model_inputs = tokenizer(sequences, max_length=max_seq_length, truncation=True)
    return model_inputs

dataset = dataset.map(preprocess, batched=True)

# Define Lora configuration
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for Lora training
model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./lora_finetuned_llama_storm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,  # Disable FP16 if using CPU
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch"
)

# Use standard data collator for language modeling
from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Set to False for causal language modeling
    pad_to_multiple_of=None
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./lora_finetuned_llama_storm")
tokenizer.save_pretrained("./lora_finetuned_llama_storm")
