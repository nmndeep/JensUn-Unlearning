import json
import re

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
model.to(device)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=device
)
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}

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


    output = pipe(messages, **generation_args)
    responses = output[0]['generated_text']
    # print(responses)

    # Extract the sentences numbered 1 through 10
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'

    matches = re.findall(pattern, responses, re.DOTALL)
    # Clean up any extra whitespace
    questions_all = [match.strip() for match in matches]
    entry = {"question": test_text} 

    # print(len(questions_all))
    for j, qq in enumerate(questions_all):
        # if j==0:
        #     print(f"Paraphrase {j+1}: {qq}")
        entry.update({f"q_phi{j+1}": qq})
        # ix+=1
    entry.update({"answer": a})
    newjson.append(entry)  # Append to list

#Save the parphrases as a json
# with open("SAVE", "w") as file:
#     json.dump(newjson, file)




