import json
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map=device
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


dataset = load_dataset('nmndeep/LKF_forget_standard', split='train')

print(len(dataset))
newjson = []
for ix, batch in enumerate(dataset):
    test_text = batch['question']
    a = batch['answer']
    messages = [
        {"role": "user", "content": f"You are a good paraphraser. I will give you a quesiton sentence, I need you to paraphrase it for me. Generate 5 grammatically correct an unique paraphrases as an enumerated list. \
        Make sure the meaning of parphrases remains the same as original question and that no new information is added. The output should be an enumerated list. \
        Question:{test_text}"}]


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,

    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'

    matches = re.findall(pattern, responses, re.DOTALL)
    # Clean up any extra whitespace
    questions_all = [match.strip() for match in matches]
    entry = {"question": test_text} 

    for j, qq in enumerate(questions_all):
        entry.update({f"q_qwen{j+1}": qq})
    entry.update({"answer": a})
    newjson.append(entry)  # Append to list

#Save the parphrases as a json
# with open("SAVE", "w") as file:
#     json.dump(newjson, file)


