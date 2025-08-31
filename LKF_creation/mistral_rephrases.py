import json
import re

import torch
from datasets import Dataset, load_dataset


device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


model.to(device)



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

	encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

	model_inputs = encodeds.to(device)

	generated_ids = model.generate(model_inputs, max_new_tokens=500, do_sample=False)
	decoded = tokenizer.batch_decode(generated_ids)
	responses = decoded[0]
	# responses = (tokenizer.decode(outputs[0], skip_special_tokens=True))

	# Extract the sentences numbered 1 through 10
	pattern = r'\d+\.\s+(.*?)(?=\n\d+\.|$)'
	matches = re.findall(pattern, responses, re.DOTALL)
	# Clean up any extra whitespace
	questions_all = [match.strip() for match in matches]
	entry = {"question": test_text} 
	print(len(questions_all))
	for j, qq in enumerate(questions_all):
		if j==0:
			print(f"Paraphrase {j+1}: {qq}")
		entry.update({f"q_mist{j+1}": qq})
		# ix+=1
	entry.update({"answer": a})
	newjson.append(entry)  # Append to list


#Save the parphrases as a json
# with open("SAVE", "w") as file:
#     json.dump(newjson, file)





