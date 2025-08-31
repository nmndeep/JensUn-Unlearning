import torch
import json
import numpy as np
import scipy
import os
from typing import List, Dict, Any
import nltk
from datasets import load_dataset

from evals.inference_helper import generate_completions


def load_rep_data():

    rep_data = load_dataset("jinzhuoran/RWKU", 'utility_fluency') # repetitiveness
    rep_data = rep_data.shuffle(seed=0)
    return rep_data



def n_gram_entropy(gen_texts:list, agg:str="arith"):
    assert agg in ["arith", "geom"]

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(
        [compute_n_gram_entropy(txt) for txt in gen_texts]
    ).item()

def compute_n_gram_entropy(sentence:str, ns:List=None, weights:List=None, agg:str="arith") -> float:
    if ns is None:
        ns = [2, 3]
    if weights is None:
        weights = [2 / 3, 4 / 3]
    assert agg in ["arith", "geom"]

    entropy_list = []
    for n in ns:
        fdist = compute_freq(sentence, n)
        freqs = np.array([freq for _, freq in fdist.items()])
        freqs = freqs / freqs.sum()
        entropy_list.append(np.sum(-freqs * np.log(freqs) / np.log(2)))
    entropy_list = np.array(entropy_list) * np.array(weights)

    return (scipy.stats.mstats.gmean if agg == "geom" else np.mean)(entropy_list)


def compute_freq(sentence, n=2):
    tokens = nltk.word_tokenize(sentence)
    ngrams = nltk.ngrams(tokens, n)
    return nltk.FreqDist(ngrams)


@torch.no_grad()
def eval_repet(model, tokenizer, dataset, batch_size:int=1, output_result_dir:str=None, use_prompt:bool=False, num_samples:int=1000):
    nltk.download('punkt')
    tokenizer.padding_side = 'left'
    prompts = []
    questions = []

    for ix, sample in enumerate(dataset['test']):
        # print(sample)
        instruction = sample['instruction']
        prompt = 'Instruction: {}\n'.format(instruction)
        if use_prompt:
            prompt = "You are an AI Assistant who is supposed to unlearn about {} " \
                     "and provide answers without its knowledge as if you never knew about it. " \
                     "Don’t tell anyone that you unlearned anything.\n".format(sample['subject']) + prompt
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(messages,
                                               tokenize=False,
                                               add_generation_prompt=True)
        prompts.append(prompt)
        questions.append(sample)
        if ix >= num_samples:
            break
    terminators = [
        [tokenizer.eos_token_id],
        [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    ]

    outputs = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=128,
        batch_size=batch_size,
        do_sample=False,
        stop_id_sequences=terminators
    )
    for answer, question in zip(outputs, questions):
        question['prediction'] = answer

    entropy = n_gram_entropy(outputs)
    print("Score {:.4f}".format(entropy*100))

    output_result = {
        'entropy': entropy*100,
        'results': questions,
    }
    tokenizer.padding_side = 'right'
    with open(output_result_dir, 'w') as f:
            json.dump(output_result, f, indent=4)
    print(f"\n✅ Saved results to: {output_result_dir}")

