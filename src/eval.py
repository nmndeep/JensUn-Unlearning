import random
import argparse
import json
import logging
import torch
import os
import sys
import json
import yaml
import argparse
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from omegaconf import OmegaConf

from evals.utils import Evaluator
from evals.judge_eval import EvalJUDGE
from evals.worst_eval import WorstEval, AvgEval
from evals.mmlu_utils import load_mmlu, eval_mmlu, log_help
from evals.eval_repetitiveness import eval_repet, load_rep_data
from evals.judge_quality import EvalWR

import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`:None for open-end generation.")

def load_model_tokenizer(cfg):
    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(cfg.model.path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # model.config.pad_token_id = tokenizer.pad_token_id

    device = cfg.model.device
    model.to(device)
    model.eval()
    return model, tokenizer

def generate_forget_responses(cfg, ev):
    dataset = load_dataset(cfg.dataset.name, split=cfg.dataset.split)
    map_fn = lambda ex: {k: ev.get_template(v) for k, v in ex.items()}

    model, tokenizer = load_model_tokenizer(cfg)

    results = []
    batch_size = cfg.generation.batch_size

    #all paraphrases per query are processed at once
    for i in tqdm(range(0, len(dataset), 1), desc="Generating"):

        batch = {k: v for k, v in dataset[i].items() if k != "answer"}
        prompts = map_fn(batch)

        ev.evaluate(model, prompts, tokenizer)
    
    assert len(ev.logs) == len(dataset), "Something is off: the output length do not match the number of questions"
    try:
        ev.save_logs()
    except Exception as e:
        raise RuntimeError(f"Failed to save logs: {e}") 

def main():
    # Load config and apply CLI overrides
    base_config = OmegaConf.load("configs/eval.yaml")
    cli_config = OmegaConf.from_dotlist(sys.argv[1:])
    cfg = OmegaConf.merge(base_config, cli_config)

    jg_file_names = []

    if cfg.output.eval_task == 'forget':
        all_evaluated = False
        while not all_evaluated:
            for icr in [False, True]:
                ev = Evaluator(cfg, icr=icr, task=cfg.output.eval_task)
                ev.logger.info(cfg)
                if not ev.load_logs_from_file()[0]:
                    generate_forget_responses(cfg, ev)
                elif not ev.load_logs_from_file()[1]:
                    ej = EvalJUDGE(cfg, task=cfg.output.eval_task, gen_file=ev.out_path, icr=icr)
                    ej.generate()
                else:
                    pass
            all_evaluated = ev.everything_evaluated()

        if all_evaluated:
            jg_file_path1 = ev.jg_file_path
            jg_file_path2 = ev.jg_file_path.replace("True", "False")

            we = WorstEval(cfg.model.name, cfg.output.task_name, cfg.output.worstdir, [jg_file_path2, jg_file_path1], task=cfg.output.eval_task)
            we.evaluate()

    elif cfg.output.eval_task == 'retain':
        
        all_evaluated = False
        while not all_evaluated:
            ev = Evaluator(cfg, icr=False, task=cfg.output.eval_task)
            ev.logger.info(cfg)
            if not ev.load_logs_from_file()[0]:
                generate_forget_responses(cfg, ev)
            elif not ev.load_logs_from_file()[1]:
                ej = EvalJUDGE(cfg, task=cfg.output.eval_task, gen_file=ev.out_path, icr=False)
                ej.generate()
            else:
                pass
            all_evaluated = ev.everything_evaluated()

        if all_evaluated:
            ae = AvgEval(cfg.model.name, cfg.output.task_name, cfg.output.worstdir, [ev.jg_file_path], task=cfg.output.eval_task)
            ae.evaluate()

    elif cfg.output.eval_task == 'mmlu':
    
        model, tokenizer = load_model_tokenizer(cfg)
        mmlu_dataset = load_mmlu()
        out_file_path = log_help('MMLU', cfg.output.gendir, cfg.output.task_name, cfg.model.name)
        eval_mmlu(model, tokenizer, mmlu_dataset, batch_size=5, output_result_dir=out_file_path, use_prompt=False, num_samples=5000)

    elif cfg.output.eval_task == 'repet':
    
        model, tokenizer = load_model_tokenizer(cfg)
        rep_dataset = load_rep_data()
        out_file_path = log_help('Rep', cfg.output.repdir, cfg.output.task_name, cfg.model.name)
        
        eval_repet(model, tokenizer, rep_dataset, batch_size=5, output_result_dir=out_file_path, use_prompt=False, num_samples=1000)
    
    elif cfg.output.eval_task == 'winrate':
        # assert cfg.output.task_name != 'pretrained', "The task_name should be for an unlearnt model"
        ew = EvalWR(cfg)
        ew.win_rate_evaluation()
        
    else:
      raise NotImplementedError("This evaluation does not exist in our protocol")


    logging.shutdown()
if __name__ == "__main__":
    main()
