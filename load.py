from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained(
        "/home/stoffi05/Downloads/DINO/"
        )

offload_dir = "./offload/"

model = AutoModelForCausalLM.from_pretrained(
        "/home/stoffi05/Downloads/DINO/",
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
        offload_state_dict=True,
        )


prompt = "Your prompt text here"

inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(**inputs)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(generated_text)
