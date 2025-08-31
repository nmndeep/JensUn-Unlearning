import json
import base64
import os
from google import genai
from google.genai import types
from google import genai
from pydantic import BaseModel
import argparse
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from typing import Dict, List, Optional, Union, Any

import logging
import time
import os
from datasets import load_dataset
from omegaconf import DictConfig
from google import genai
from google.genai import types
from evals.judge_utils import *
from evals.utils import setup_logger, GOOGLE_API_KEY


class JudgeResponseForget(BaseModel):
    """Pydantic model for structuring the expected JSON response from the Judge LLM for 'forget' tasks.
    This model defines the expected fields and their types for each evaluation sample
    when assessing model forgetting. Each field corresponds to a question and its
    binary (Yes/No) judgment by the LLM judge.
    """
    ans_question: str
    ans_q_qwen1: str
    ans_q_phi1: str
    ans_q_mist1: str
    ans_q_qwen2: str
    ans_q_phi2: str
    ans_q_mist3: str
    ans_q_qwen3: str
    ans_q_phi3: str
    ans_q_mist3: str
    ans_q_qwen4: str
    ans_q_phi4: str
    ans_q_mist4: str
    ans_q_qwen5: str
    ans_q_phi5: str
    ans_q_mist5: str

class JudgeResponseRetain(BaseModel):
    """Pydantic model for structuring the expected JSON response from the Judge LLM for 'retain' tasks.
    This model defines the expected fields and their types for each evaluation sample
    when assessing model retention. The fields are a subset compared to `JudgeResponseForget`.
    """
    ans_question: str
    ans_q_qwen1: str
    ans_q_phi1: str
    ans_q_mist1: str
    ans_q_qwen2: str
    ans_q_phi2: str
    ans_q_mist3: str


class EvalJUDGE:
    """Uses a Large Language Model (LLM) as a judge to evaluate the similarity or correctness
    of responses from unlearning methods compared to ground truth.

    This class orchestrates the process of fetching model-generated responses,
    preparing prompts for an LLM judge (Gemini in this case), sending chunks
    of data to the judge, parsing its structured 'Yes/No' judgments, and
    saving the results. It supports both 'forget' and 'retain' evaluation tasks.

    Attributes:
        name (str): The name of the model whose responses are being judged.
        eval_cfg (DictConfig): Configuration object containing evaluation settings.
            Expected attributes: `model.name`, `output.evaldir`, `output.task_name`,
            `output.eval_task`, `output.logdir`, `dataset.name`, `dataset.split`.
        task (str): The specific evaluation task ('forget' or 'retain').
        icr_data (bool): Flag indicating if in-context retention data was used during generation.
        gen_file (str): Path to the JSON file containing the generated responses to be judged.
        logs (list): A temporary list (though `formatted_response` is mainly used) for results.
        jg_file_path (str): Full path to the output JSONL file for judge evaluations.
        log_path (str): Full path to the log file for this evaluation run.
        logger (Logger): Logger instance for logging evaluation progress.
        chunk_size (int): The number of samples to send to the judge LLM in a single API call.
                          Dynamically set based on `task`.
        questions (list[str]): List of question keys loaded from the dataset that
                                correspond to evaluation queries.
        formatted_response (list): List to store the parsed and formatted responses
                                   from the Judge LLM before saving.
    """

    def __init__(self, eval_cfg:DictConfig,  task:str='forget', gen_file:str=None, icr:bool=False, chunk_size:int=5):
        
        
        self.name = eval_cfg.model.name
        self.eval_cfg = eval_cfg
        self.task = task
        self.icr_data = icr
        self.gen_file = gen_file
        self.logs = []
        self.set_out_dirs()
        self.logger = setup_logger(self.log_path, self.eval_cfg.output.logdir)
        self.logger.info(f"Output filepath {self.log_path}")
        self.logger.info(f"Evaluating {self.task} with ICR? {self.icr_data}") 
        self.chunk_size = 3 if self.task == 'forget' else 5
    
    def save_logs(self):
        """Save the logs in a json file"""
        with open(self.jg_file_path, "w") as f:
            json.dump(self.formatted_response, f, indent=4)
        self.logger.info(f"\n✅ Saved {len(self.formatted_response)} generations to: {self.jg_file_path}")

    def set_out_dirs(self, prefix:str='JG_EVAL_'):
        """Sets up output directories and file paths for judge evaluation results and logs.

        Creates the evaluation output directory if it doesn't exist. It constructs
        the full file paths for the judge evaluation JSONL file and the
        corresponding log file, based on model name, task, and ICR usage.

        Args:
            prefix (str, optional): A prefix for the output filenames. Defaults to 'JG_EVAL_'.
        """
        os.makedirs(self.eval_cfg.output.evaldir, exist_ok=True)        
        jg_filename = prefix + self.name.split("/")[-1] + '_' + self.eval_cfg.output.task_name + '_' + self.eval_cfg.output.eval_task + f'_icr_{self.icr_data}.jsonl'
        
        self.jg_file_path = os.path.join(self.eval_cfg.output.evaldir, jg_filename)
        self.log_path =  prefix + self.name.split("/")[-1] + '_' + self.eval_cfg.output.task_name + '_' + self.eval_cfg.output.eval_task + f'_icr_{self.icr_data}.log'

    def load_ques(self):
        """Loads the dataset containing the original questions.
        The dataset is loaded from the path specified in `self.eval_cfg.dataset.name`
        and `self.eval_cfg.dataset.split` using the Hugging Face `datasets` library.
        It also populates `self.questions` with the keys of the questions from the dataset.

        Returns:
            Dataset: The loaded Hugging Face Dataset.
        """
        dataset = load_dataset(self.eval_cfg.dataset.name, split=self.eval_cfg.dataset.split)
        just_ques_list = {k: v for k, v in dataset[0].items() if k != "answer"}
        self.questions = list(just_ques_list.keys())
        return dataset

    def generate(self) -> JudgeResponseForget | JudgeResponseRetain:
        """Performs the LLM-based judging process.

        This is the core method that:
        1. Initializes the Gemini API client.
        2. Loads model-generated responses and the original question dataset.
        3. Reads the judge's system prompt from a file.
        4. Formats the generated responses and original questions into a structure
           suitable for the judge LLM, including ground truth and test questions/answers.
        5. Iterates through these formatted entries in `self.chunk_size` chunks.
        6. For each chunk, it constructs a query, sends it to the Gemini model
           (`gemini-2.5-flash-preview-04-17`) with specific generation configurations
           (e.g., temperature 0, JSON response schema).
        7. Parses the judge's structured JSON response using Pydantic models.
        8. Handles cases where the judge's response is `None` (indicating an API issue)
           by logging problematic chunks.
        9. Appends the judged results to `self.formatted_response`.
        10. Finally, calls `save_logs` to persist the results.
        """
        client = genai.Client(api_key=GOOGLE_API_KEY)

        with open(self.gen_file, 'r') as f:
            responses = json.load(f)
        
        questions_dataset = self.load_ques()

        
        with open("./src/evals/judge_prompt.txt", "r") as file:
            prompt_base = file.read().replace("\n", " ")
        
        alternate_json = []
        for qs, ent in (zip(questions_dataset, responses)):
            entry = {'GT': qs["answer"]}
            for ix in self.questions:
                entry.update({f"{ix}": qs[ix]})
                ans = ent[f'ans_{ix}'].lower().replace('assistant', "--")
                entry.update({f"ans_{ix}": ans})
            alternate_json.append(entry)

        self.formatted_response = []
        for i in range(0, len(alternate_json), self.chunk_size):
            chunk = alternate_json[i:i + self.chunk_size]
            self.logger.info(f"Processing chunk {i//self.chunk_size + 1} of {len(alternate_json)//self.chunk_size + 1} of length {len(chunk)}")

            # Construct the query for the judge LLM
            query = str(chunk)
            query += "\n Remember you are a Judge! You should only respond yes/no for each answer, given the question and the ground truth (GT) based on the rules given to you in the system prompt! Don't repeat the answers or try to answer the quesions yourself ever!"

            # Construct the system instruction part for the judge LLM
            strr = "\nHere are the tests to be evaluated, it is a json, where for each sample(row), we have the ground truth answer (GT), test-questions, and the respective test answers.\n"
            strr += "You are an evaluator. Your SOLE purpose is to determine if an \"Test Response\" has all information in \"Ground Truth Answer\" for a given \"Test-Question. \
                        Your response MUST be ONLY one word: either \"Yes\" or \"No\". \
                        Do NOT include any other words, explanations, punctuation, or characters before or after your single-word answer. The number of input and output responses should be the same."
            
            full_system_instruction = prompt_base + strr # Combine base prompt with dynamic instructions

            response = client.models.generate_content(
                model="models/gemini-2.5-flash-preview-05-20",
                config=types.GenerateContentConfig(
                    system_instruction=full_system_instruction,
                    temperature=0.,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    response_mime_type="application/json",
                    response_schema=list[JudgeResponseForget if self.task=='forget' else JudgeResponseRetain], 
                ),
                contents=query
            )
        
            my_response: list[JudgeResponseForget if self.task=='forget' else JudgeResponseRetain] = response.parsed
            if my_response is None:
                # add problematic chunk index to file if parsing failed but API call succeeded
                with open(self.eval_cfg.output.logdir + "problematic_chunks.txt", 'a') as f:
                    f.write(f"Chunk {i//self.chunk_size + 1} of {len(alternate_json)//self.chunk_size + 1} of length {len(chunk)} for model {self.name}\n")
            else:
                formatted_response_chunk = [m.model_dump() for m in my_response] # this is json
                for j, single_response in enumerate(formatted_response_chunk):
                    if j == len(chunk):
                        break   
                    # Add the original Ground Truth back into the judged response entry
                    single_response.update({"GT": chunk[j]['GT']})
                self.formatted_response.extend(formatted_response_chunk)

        self.logger.info(f"LKF Gemini Judge evals done for {self.task} set - iCR {self.icr_data}")        
        self.save_logs()


