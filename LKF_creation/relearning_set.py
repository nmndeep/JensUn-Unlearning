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
newjson = []

topics= ['history', 'general knowledge', 'geography', 'biology', 'physics', 'chemistry', 'art', 'europe',
		'Asia', 'Australia', 'Africa', 'Soccer', 'Basketball', 'Cricket', 'Language', 'finance', 'maths', 
		'track and field', 'cooking', 'clothing', 'winter', 'summer']

for ix, sampl in enumerate(topics):

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant.",
        "role": "user", "content": f"Create 10 highly specific and easy question-answer pairs about general facts about the topic: {sampl}. \
        The questions and answers should be self-contained and not need any reference, i.e. \
		every question should clearly indicate that it is about {sampl}. The answer should be short, either one word, or at most a few, the questions answers should be numbered 1-10."}]

    output = pipe(messages, **generation_args)
    responses = output[0]['generated_text']

    # Extract the sentences numbered 1 through 10
    pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'

    matches = re.findall(pattern, responses, re.DOTALL)
    # Clean up any extra whitespace
    questions_all = [match.strip() for match in matches]

    for j, qq in enumerate(questions_all):
        entry = {}
        parts = qq.split('\n')
        question = parts[0].replace("Question: ", "").strip()
        answer = parts[1].replace("Answer: ", "").strip()
        entry.update({f"question": question})
        # ix+=1
        entry.update({"answer": answer})
        newjson.append(entry)  # Append to list
    if ix == 0:
        print(newjson)

#Save teh json, extract to dataset and push to hub
# with open("SAVE", "w") as file:
#     json.dump(newjson, file)








