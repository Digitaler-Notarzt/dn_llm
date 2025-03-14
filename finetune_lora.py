from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM
from transformers import Trainer

model_name = "akjindal53244/Llama-3.1-Storm-8B"
max_seq_length = 2048


# Define quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True  # Enable FP32 offloading to CPU
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_memory = {
    "cpu": "100GiB",
    str(device): "12GiB"  # Adjust based on your GPU's VRAM
}

# Define a custom device map for offloading model components
device_map = "auto"  # Automatically distribute layers between CPU and GPU

# Load the model with quantization and offloading
model = AutoModelForCausalLM.from_pretrained(
   "akjindal53244/Llama-3.1-Storm-8B",
   device_map=device_map,
    quantization_config=quantization_config,
    max_memory=max_memory
)

tokenizer = AutoTokenizer.from_pretrained("akjindal53244/Llama-3.1-Storm-8B")


dataset = load_dataset("json", data_files="trainData.json")

lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = model.prepare_for_kbit_training(use_gradient_checkpointing=True)
model = FastLanguageModel.get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./lora_finetuned_llama_storm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    logging_steps=10,
    optim="adamw_torch"
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=FastLanguageModel.get_unsloth_data_collator(tokenizer)
)

trainer.train()

model.save_pretrained("./lora_finetuned_llama_storm")
tokenizer.save_pretrained("./lora_finetuned_llama_storm")
