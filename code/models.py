# 5. models.py
# Wraps HuggingFace LLaMA-2 or GPT-3.5 (if API available).

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_model(model_name="meta-llama/Llama-2-7b-chat-hf"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    return model, tokenizer

def run_llm(model, tokenizer, prompt, max_new_tokens=128):
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)
