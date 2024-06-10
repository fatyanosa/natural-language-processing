import torch
from peft import PeftModel, LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline,BitsAndBytesConfig
)


# model = "Ichsan2895/Merak-7B-v2"  # path/to/your/model/or/name/on/hub"
model = "fatyanosa/Merak-7B-squadpairs-indo"

# Hugging face repo name
# model = "meta-llama/Llama-2-7b-chat-hf" #chat-hf (hugging face wrapper version)

tokenizer = AutoTokenizer.from_pretrained(model)

pipe = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto" # if you have GPU
)

# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     top_p = 0.9,
#     temperature = 0.2,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200, # can increase the length of sequence
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

# Run text generation pipeline with our model
prompt = "Jelaskan mengenai pembelajaran mesin?"
instruction = f"### Instruction:\n{prompt}\n\n### Response:\n"
# pipe = pipeline(
#     task="text-generation", model=model, tokenizer=tokenizer, max_length=128
# )
result = pipe(instruction)
print(result[0]["generated_text"][len(instruction) :])
