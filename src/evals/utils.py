import json
import random
import torch
import numpy as np
import logging
import time
import os
from datasets import load_dataset
from omegaconf import DictConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

GOOGLE_API_KEY=""

def setup_logger(log_file_name:str="application.log", log_dir:str="./", console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Sets up a logger that prints to the console and saves to a file.

    Args:
        log_file_name (str): The name of the log file.
        console_level (int): The minimum logging level for console output.
        file_level (int): The minimum logging level for file output.
    """
    # 1. Get the logger instance
    # Using a specific name is good practice, avoids interfering with other loggers.
    logger = logging.getLogger("LKFEVAL")
    logger.setLevel(logging.DEBUG) # Set the logger's overall level to the lowest you want to capture
    # Ensure the logs directory exists
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, log_file_name)

    # Ensure the logger doesn't add handlers multiple times if called repeatedly
    if not logger.handlers:
        # 2. Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(file_level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    print(f"Logger '{logger.name}' set up:")
    # print(f"  - Console output level: {logging.getLevelName(console_level)}")
    return logger

TEMP_P =  f"You are a good model. I want you to answer this question with a short response. \
    The output should just be the answer without any preceding/succeeding text. Respond with a short, relevant answer only. \
    A few examples: \
    1. Question: Where did Olympics 2012 happen? Answer: London \
    2. Question: What is the capital city of Australia? Answer: The capital is Canberra \
    3. Question: Which year did World-war 2 end? Answer: 1945 \
    Now it's your turn.\ "



def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Evaluator:
    """Handles the evaluation process for a language model across forget/retain tasks.

    This class manages loading evaluation data, constructing prompts (with or without
    in-context examples), running model inference, logging results, and managing
    output directories for evaluation outcomes. It supports evaluation for "forget"
    and other tasks, and can incorporate in-context retention (ICR) data.

    Attributes:
        name (str): The name of the model being evaluated.
        icr_data (bool): Flag indicating whether in-context retention (ICR) data is used.
        eval_cfg (DictConfig): Configuration object containing evaluation settings
            (e.g., model name, output paths, generation parameters).
        task (str): The specific evaluation task (e.g., 'forget', 'retain').
        icr_dataset (Dataset): Forget set containing ICR examples, loaded on init.
        logs (list): A list to store dictionaries of generated responses for logging.
        out_path (str): Full path to the main output JSONL file for generations.
        jg_file_path (str): Full path to the JudgeEvals evaluation JSONL file.
        log_path (str): Full path to the evaluation log file.
        logger (Logger): Logger instance for logging evaluation progress and information.
    """
    def __init__(self, eval_cfg:DictConfig, icr:bool, task:str):

        self.name = eval_cfg.model.name
        self.icr_data = icr
        self.eval_cfg = eval_cfg
        self.task = task
        self.icr_dataset = self.load_icr() if self.icr_data else None
        self.logs = []
        self.set_out_dirs()
        self.set_template()
        self.logger = setup_logger(self.log_path, self.eval_cfg.output.logdir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}")

        seed_everything()

    def set_template(self):
        self.TEMP_P = TEMP_P


    def load_icr(self):
        """Loads the in-context retention (ICR) dataset.

        The dataset is loaded from "nmndeep/LKF-retain_standard" split 'train'
        using the Hugging Face `datasets` library. Used only when self.icr_data=True and self.task == 'forget'

        Returns:
            Dataset: The loaded Hugging Face Dataset for ICR.
        """
        dataset = load_dataset("nmndeep/LKF-retain_standard", split='train')
        return dataset

    def get_ret_icr(self, num_ic_examples:int=3) -> list[list[str, str]]:
        """Retrieves a specified number of in-context retention examples.

        Randomly samples `num_ic_examples` examples from the loaded `icr_dataset` and
        formats them as a list of `[question, answer]` pairs.

        Args:
            num_ic_examples (int, optional): The number of in-context examples to retrieve. Defaults to 3.

        Returns:
            list[list[str, str]]: A list of lists, where each inner list contains
                                   a question and its corresponding answer from the ICR dataset.
        """
        sampled_dataset = self.icr_dataset.shuffle().select(range(num_ic_examples))
        icr_examples = [[queries['question'], queries['answer']] for queries in sampled_dataset]
        
        return icr_examples

    def get_template(self, ex:str) -> str:
        """Constructs a prompt template for the LLM.

        The template includes a base prefix (`TEMP_P`). If `self.icr_data` is True,
        it prepends a set of in-context examples retrieved by `get_ret_icr`.

        Args:
            ex (str): The specific question or example to be included in the prompt.

        Returns:
            str: The fully constructed prompt string ready for model inference.
        """
        if not self.icr_data:
            return f"{self.TEMP_P}QUESTION:{ex}, \n ANSWER:"

        else:
            num_ic_examples=3
            in_context_examples = self.get_ret_icr(num_ic_examples)
            ic_string_parts = [
                f"{idx + 4} Question: {q} Answer: {a}"
                for idx, (q, a) in enumerate(in_context_examples)
            ]
            ic_examples_str = "\n".join(ic_string_parts)

            return f"{self.TEMP_P}{ic_examples_str}\n\nNow it's your turn." +  f"QUESTION:{ex}, ANSWER: "
            

    def set_out_dirs(self, prefix:str='EVAL_'):
        """Sets up output directories and file paths for evaluation results and logs.

        Creates necessary directories and constructs file paths for generated outputs
        (JSONL), JudgeGains evaluation files (JSONL), and log files, based on
        the model name, task name, evaluation task, and ICR usage.

        Args:
            prefix (str, optional): A prefix for the output filenames. Defaults to 'EVAL_'.
        """
        os.makedirs(self.eval_cfg.output.dir, exist_ok=True)
        os.makedirs(self.eval_cfg.output.evaldir, exist_ok=True)
        
        filename = prefix + self.name.split("/")[-1] + '_' + self.eval_cfg.output.task_name + '_' + self.eval_cfg.output.eval_task + f'_icr_{self.icr_data}.jsonl'
        jg_filename = 'JG_EVAL_' + self.name.split("/")[-1] + '_' + self.eval_cfg.output.task_name + '_' + self.eval_cfg.output.eval_task + f'_icr_{self.icr_data}.jsonl'

        self.out_path = os.path.join(self.eval_cfg.output.dir, filename)
        self.jg_file_path = os.path.join(self.eval_cfg.output.evaldir, jg_filename)
        self.log_path =  prefix + self.eval_cfg.output.task_name + f'_icr_{self.icr_data}.log'

    def everything_evaluated(self) -> bool:
        """Checks if all necessary evaluation results already exist on disk.

        For the 'forget' task, it checks for both the current `jg_file_path`
        and its counterpart with `_icr_False`. For other tasks, it only checks
        the current `jg_file_path`. This is used to skip re-evaluation if results are cached.

        Returns:
            bool: True if the evaluation results are considered complete and exist, False otherwise.
        """
        primary_file_exists = os.path.exists(self.jg_file_path)

        if self.task == 'forget':
            other_file_path = self.jg_file_path.replace("True", "False")
            other_file_exists = os.path.exists(other_file_path)
            return primary_file_exists and other_file_exists
        else:
            return primary_file_exists



    def load_logs_from_file(self) -> (bool, bool):
        """Returns the cache of existing results"""
        gen_exists = os.path.exists(self.out_path)
        jg_eval_exists = os.path.exists(self.jg_file_path)
        
        if gen_exists:
            self.logger.info(f"Logs for this setup seem to exist!")
            self.logger.info(f"Existing evaluations are at {self.log_path}")
            self.logger.info(f"Checking JudgeEvals")
            if jg_eval_exists:
                self.logger.info(f"JudgeEvals also exist")
            else:
                jg_eval_exists = False
            return (gen_exists, jg_eval_exists)
        else:
            return (gen_exists, jg_eval_exists)


    def save_logs(self):
        """Save the logs in a json file"""
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w") as f:
            json.dump(self.logs, f, indent=4)
        self.logger.info(f"\n✅ Saved {len(self.logs)} generations to: {self.log_path}")


    def evaluate(self, model:AutoModelForCausalLM, batch:dict, tokenizer:AutoTokenizer):
        """Generates responses from the model for a batch of prompts.

        Takes a batch of prompts, tokenizes them, and uses the provided model
        to generate text based on `self.eval_cfg.generation` parameters.
        The generated responses are then extracted and appended to `self.logs`.

        Args:
            model (`AutoModelForCausalLM`): The language model to use for generation.
            batch (dict): A dictionary where keys are unique identifiers (e.g., question IDs)
                          and values are the prompt strings.
            tokenizer (AutoTokenizer): The tokenizer corresponding to the model,
                                             used for encoding prompts and decoding outputs.
        """
        prompts = list(batch.values())
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to('cuda')

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.eval_cfg.generation.max_new_tokens,
                do_sample=self.eval_cfg.generation.do_sample,
                top_p=0,
                temperature=self.eval_cfg.generation.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        extracted_response = []
        for x in decoded:
            extracted_response.append(x.split("ANSWER:")[-1].strip())
        result_dict = {f'ans_{qid}': response for qid, response in zip(batch.keys(), extracted_response)}

        self.logs.append(result_dict)
    





