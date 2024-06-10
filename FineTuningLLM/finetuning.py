import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoTokenizer,
    pipeline,
)

from peft import LoraConfig, PeftModelForCausalLM, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# Model
base_model = "Ichsan2895/Merak-7B-v2"
new_model = "Merak-7B-squadpairs-indo"

# Dataset
dataset = load_dataset("fatyanosa/squad_pairs_indo_1000", split="train")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# LoRA configuration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
)

# Load base moodel
model = AutoModelForCausalLM.from_pretrained(
    base_model, quantization_config=bnb_config, device_map={"": 0}
)

# Cast the layernorm in fp32, make output embedding layer require grads, add the upcasting of the lmhead to fp32
model = prepare_model_for_kbit_training(model)

# Set training arguments
training_arguments = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    evaluation_strategy="steps",
    eval_steps=1000,
    logging_steps=1,
    optim="paged_adamw_8bit",
    learning_rate=2e-4,
    lr_scheduler_type="linear",
    warmup_steps=10,
    report_to="wandb",
    max_steps=2,  # Remove this line for a real fine-tuning
)


# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    eval_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="0",
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Train model
trainer.train()

# Save trained model
trainer.model.save_pretrained(new_model)

# Store New Llama2 Model
# Reload model in FP16 and merge it with LoRA weights
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map={"": 0},
)

model = PeftModelForCausalLM.from_pretrained(model, new_model)
model = model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

access_token_write = "token"
model.generation_config.temperature = None
model.generation_config.top_p = None
model.push_to_hub(new_model, use_temp_dir=False, token=access_token_write, check_pr=True)
tokenizer.push_to_hub(new_model, use_temp_dir=False, token=access_token_write, check_pr=True)