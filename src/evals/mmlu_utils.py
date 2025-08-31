import json
import os

import numpy as np
import torch
from datasets import load_dataset
from evals.inference_helper import get_next_word_predictions

choices = ["A", "B", "C", "D"]

def log_help(prefix:str, dirr:str, task_name:str, modelname:str) -> str:

    os.makedirs(dirr, exist_ok=True)
    filename = f"{prefix}_" + modelname.split("/")[-1] + '_' + task_name + '.jsonl'
    out_path = os.path.join(dirr, filename)
    return out_path

def load_mmlu():
    utility_general = load_dataset("jinzhuoran/RWKU", 'utility_general') # general ability
    utility_general = utility_general.shuffle(seed=0)
    return utility_general

def format_example(sample:dict, include_answer:bool=True) -> str:
    prompt = sample['question']
    prompt = 'Question: ' + prompt
    for j in range(len(sample['choices'])):
        prompt += "\n{}. {}".format(choices[j], sample['choices'][j])
    if include_answer:
        prompt += "\nAnswer:"
        prompt += " {}\n\n".format(choices[sample['answer']])
    return prompt


def gen_prompt(dev_set:list, subject:str) -> str:
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        subject.replace('_', ' ')
    )
    for sample in dev_set:
        prompt += format_example(sample)
    return prompt


@torch.no_grad()
def eval_mmlu(model, tokenizer, dataset, batch_size=1, output_result_dir=None, use_prompt=False, num_samples=5000):
    tokenizer.padding_side = 'left'
    prompts = []
    dataset = dataset['test']
    for ix, sample in enumerate(dataset):
        dev_set = sample['examples']
        subject = sample['task']
        prompt_end = format_example(sample, include_answer=False)
        train_prompt = gen_prompt(dev_set, subject)
        prompt = train_prompt + 'Please following the previous examples and answer the given question.\n' + prompt_end
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        prompt += "Answer:"
        prompts.append(prompt)
        if ix >= num_samples:
            break
    answer_choice_ids = [tokenizer.encode(" " + answer_choice, add_special_tokens=False)[-1] for answer_choice in
                         choices]
    pred_indices, all_probs = get_next_word_predictions(
        model, tokenizer, prompts, candidate_token_ids=answer_choice_ids, return_token_predictions=False,
        batch_size=batch_size
    )

    # get the metrics
    cors = []
    ans, preds = [], []
    for i in range(len(pred_indices)):
        prediction = pred_indices[i]
        ground_truth = dataset[i]['answer']
        dataset[i]['prediction'] = prediction
        cors.append(prediction == ground_truth)
        ans.append(ground_truth)
        preds.append(prediction)

    acc = np.mean(cors)
    print("Average accuracy:  {:.4f}".format(acc))
    tokenizer.padding_side = 'right'
    output_result = {
        'acc': np.mean(cors),
        'all_acc': cors,
        'GT': ans,
        'outputs': preds,
    }

    with open(output_result_dir, 'w') as f:
        json.dump(output_result, f, indent=4)
    print(f"\n✅ Saved results to: {output_result_dir}")
