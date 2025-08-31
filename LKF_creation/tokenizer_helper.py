import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
print("EOS Token:", tokenizer.eos_token)
print("EOS Token ID:", tokenizer.eos_token_id)
list_ans = ["#", "No idea", " " , ","]
for text in list_ans:
    print(f"Text.  : {text}")
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    print(token_ids)
    
    decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
    
    for i, (tok, decoded) in enumerate(zip(tokens, decoded_tokens)):
        print(f"Token {i}: '{tok}' -> '{decoded}'")
