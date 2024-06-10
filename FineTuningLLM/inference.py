import torch
from peft import PeftModelForCausalLM, LoraConfig, get_peft_model, get_peft_config
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,BitsAndBytesConfig
)


model_name  = "Ichsan2895/Merak-7B-v2"  # path/to/your/model/or/name/on/hub"
adapter = "fatyanosa/Merak-7B-squadpairs-indo"

# Hugging face repo name
# model = "meta-llama/Llama-2-7b-chat-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
else:
  compute_dtype = torch.float16

config = {
    "r":16,
    "lora_alpha":32,
    "lora_dropout":0.05,
    "bias":"none",
    "task_type":"CAUSAL_LM",
    "peft_type":"LORA",
    "target_modules":[
        "up_proj",
        "down_proj",
        "gate_proj",
        "k_proj",
        "q_proj",
        "v_proj",
        "o_proj",
    ],
}

peft_config = get_peft_config(config)

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
          model_name, quantization_config=bnb_config, torch_dtype=compute_dtype, device_map={"": 0}
)
model = PeftModelForCausalLM(model, peft_config)
model.print_trainable_parameters()

pipe = pipeline(
    "text-generation",
    model=model_name,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

# Run text generation pipeline with our model
prompt = "Jelaskan mengenai pembelajaran mesin?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
# pipe = pipeline(
#     task="text-generation", model=model, tokenizer=tokenizer, max_length=128
# )
result = pipe(instruction)
print(result[0]["generated_text"][len(instruction) :])
