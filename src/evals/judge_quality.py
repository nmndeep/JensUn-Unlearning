import json
import os
import re
import argparse
from typing import List, Dict, Any

from dataclasses import dataclass
from pydantic import BaseModel
from tqdm import tqdm
from omegaconf import DictConfig
from google.genai import types

from google import genai
from evals.utils import GOOGLE_API_KEY




# Define a dataclass for the structured response from the judge model
@dataclass
class JudgeResponse:
    """
    Represents a structured response from a judge model.

    Attributes:
        score_assistant_1 (int): The score given to assistant 1.
        score_assistant_2 (int): The score given to assistant 2.
        explanation (str): A textual explanation for the scores.
    """
    score_assistant_1: int
    score_assistant_2: int
    explanation: str


class EvalWR:
    """
    Manages the evaluation workflow for comparing two assistants (or models).

    This class is responsible for setting up evaluation parameters,
    managing output directories, and preparing prompts for the evaluation tasks.
    It encapsulates the configuration and setup logic required before
    running an evaluation.

    Attributes:
        name (str): The name of the model being evaluated, derived from `eval_cfg.model.name`.
        eval_cfg (DictConfig): The evaluation configuration object, containing details
                  about the models, tasks, and output settings.
        task_name (str): The name of the evaluation task, derived from `eval_cfg.output.task_name`.
        nsamples (int): The number of samples to be used for evaluation,
                        defaulting to 100.
        out_dir (str): The base output directory for evaluation results. (Inferred from set_out_dirs)
        results_dir (str): The directory for storing evaluation results. (Inferred from set_out_dirs)
        prompts_dir (str): The directory for storing generated prompts. (Inferred from set_out_dirs)
    """

    def __init__(self, eval_cfg:DictConfig):
        
        self.name = eval_cfg.model.name
        self.eval_cfg = eval_cfg
        self.task_name = eval_cfg.output.task_name
        self.nsamples = 100
        self.set_out_dirs()
        self.prompts()

    def set_out_dirs(self, prefix:str='WR_'):
        """Sets up output directories and file paths for judge evaluation results and logs.

        Args:
            prefix (str, optional): A prefix for the output filenames. Defaults to 'JG_EVAL_'.
        """
        os.makedirs(self.eval_cfg.output.windir, exist_ok=True)        
        self.repfile_base = os.path.join(self.eval_cfg.output.repdir, f"Rep_" + self.name.split("/")[-1] + '_' + 'pretrained' + '.jsonl')
        assert os.path.exists(self.repfile_base), "The AlpacaEval generations do not yet exist for the pretrained basemodel; do `eval_task=repet` first for task_name=pretrained"
        self.repfile_unlearnt = os.path.join(self.eval_cfg.output.repdir, f"Rep_" + self.name.split("/")[-1] + '_' + f'{self.task_name}' + '.jsonl')
        assert os.path.exists(self.repfile_unlearnt), "The AlpacaEval generations do not yet exist for the unlearnt model; do `eval_task=repet`"    
        out_filename = prefix + self.name.split("/")[-1] + '_' + self.task_name + '.jsonl'
        self.out_file_path = os.path.join(self.eval_cfg.output.windir, out_filename)
    
    def prompts(self):
        self.sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer. Your response MUST be a JSON object conforming to the following structure: {'score_assistant_1': int, 'score_assistant_2': int, 'explanation': str}."
        self.prompt_template = "[Question]\n{question}\n\n[The Start of Assistant 1's Answer]\n{answer_1}\n\n[The End of Assistant 1's Answer]\n\n[The Start of Assistant 2's Answer]\n{answer_2}\n\n[The End of Assistant 2's Answer]\n\n[System]\n{criteria}\n\n"
        self.criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nProvide the scores for Assistant 1 and 2, and a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment, all within the specified JSON format."

    
    def call_gemini_api(self, full_prompt:str) -> list[JudgeResponse]:
        """
        Calls the Gemini API using the google.generativeai client.
        It expects a structured JSON response based on the JudgeResponse schema.
        """
        if GOOGLE_API_KEY:
            client = genai.Client(api_key=GOOGLE_API_KEY)
        else:
            # If API_KEY is not provided, assumes the environment handles authentication
            # For Canvas, you would rely on __initial_auth_token and `genai.get_default_configured_client()`
            # or similar environment-specific setup.
            print("API_KEY not provided. Assuming environment handles authentication.")
            # If running outside a Google-managed environment, you must provide an API_KEY.

        query = [{"role": "user", "parts": [{"text": full_prompt}]}]

        try:
            response = client.models.generate_content(
                model="models/gemini-2.5-flash-preview-05-20",
                config=types.GenerateContentConfig(
                    system_instruction=self.sys_prompt, # Use the system prompt defined globally
                    temperature=0.0, # Setting temperature to 0 for deterministic output
                    # thinking_config=types.ThinkingConfig(thinking_budget=0), # Removed as it's not a standard parameter for generate_content
                    response_mime_type="application/json",
                    response_schema=list[JudgeResponse], # Expecting a list of JudgeResponse objects
                ),
                contents=query
            )

            # The response.text will now be a JSON string if response_mime_type is honored
            if response and response.text:
                # Parse the JSON string from the model's response
                parsed_response = json.loads(response.text)
                # Assuming the model returns a list with a single JudgeResponse object
                if parsed_response and isinstance(parsed_response, list) and len(parsed_response) > 0:
                    judge_response = parsed_response[0] # Get the first (and likely only) object
                    return judge_response
                else:
                    print(f"Warning: Unexpected JSON structure received: {parsed_response}")
                    return None
            return None

        except Exception as e:
            print(f"API request failed: {e}")
            # Re-raise to be caught by the main loop's try-except
            raise


    def calculate_win_rate(self, evaluations_data:List[Dict[Any, Any]]) -> (float, dict):
        """
        Calculates the win rate of assistant_2 against assistant_1 (reference).

        Args:
            evaluations_data (list of dict): A list of evaluation dictionaries,
                                             each containing 'score_assistant_1' and
                                             'score_assistant_2'.

        Returns:
            float: The win rate of assistant_2.
            dict: A dictionary containing the counts of wins, losses, and ties for assistant_2.
        """
        wins = 0
        losses = 0
        ties = 0
        total_comparisons = len(evaluations_data)

        if total_comparisons == 0:
            return 0.0, {"wins": 0, "losses": 0, "ties": 0}

        for evaluation in evaluations_data:
             score_a1 = evaluation['score_assistant_1']
             score_a2 = evaluation['score_assistant_2']
             if isinstance(score_a1, int) and isinstance(score_a2, int):
               if score_a2 > score_a1:
                   wins += 1
               elif score_a2 < score_a1:
                   losses += 1
               else:
                   ties += 1
             else:
                pass
        # In win-rate calculation, ties are typically split, or sometimes 0.5 is given to each.
        # For a straightforward win-rate, we'll count actual wins.
        # If you want to include ties (e.g., 0.5 for each), modify the calculation.
        win_rate = (wins + 0.5 * ties) / total_comparisons

        return win_rate, {"win-rate": win_rate, "wins": wins, "losses": losses, "ties": ties}


    def win_rate_evaluation(self):

        if not os.path.isfile(self.out_file_path):
            data_pairs = []

            with open(self.repfile_base , "r") as f:
              data1 = json.load(f)['results']
            with open(self.repfile_unlearnt , "r") as f:
              data2 = json.load(f)['results']

            ct = 0
            for (item1, item2) in zip(data1, data2):
                # print(item2)
                data_pairs.append({
                "question": f"{item1['instruction']}. (Pair {ct+1})",
                "answer_1": f"{item1['prediction']}. (Assistant 1, Pair {ct+1})",
                "answer_2": f"{item2['prediction']}. (Assistant 2, Pair {ct+1})"
                })
                ct += 1
                if ct >= self.nsamples: 
                    break

            # --- Main Evaluation Logic ---
            evaluation_results = []

            print(f"Starting evaluation of {len(data_pairs)} question-answer pairs... for unlearn-model against {self.name}")

            for i, pair in enumerate(tqdm(data_pairs)):
                question = pair["question"]
                answer_1 = pair["answer_1"]
                answer_2 = pair["answer_2"]
                # Construct the full prompt (excluding sys_prompt which is now in config)
                # The criteria includes the instruction for JSON output, so it's part of the user prompt
                full_prompt = self.prompt_template.format(
                  question=question,
                  answer_1=answer_1,
                  answer_2=answer_2,
                  criteria=self.criteria
                )
                try:
                  # Call the Gemini API
                  judge_response_obj = self.call_gemini_api(full_prompt)

                  if judge_response_obj:
                      score_assistant_1 = judge_response_obj.get('score_assistant_1')
                      score_assistant_2 = judge_response_obj.get('score_assistant_2')
                      explanation = judge_response_obj.get('explanation')
                  else:
                      score_assistant_1 = None
                      score_assistant_2 = None
                      explanation = "Failed to get structured response from API."
                      print(f"Error: No structured response for question: {question}")

                except Exception as e:
                  # Catch any errors during API call or parsing
                  print(f"An error occurred while processing question: {question}. Error: {e}")
                  score_assistant_1 = None
                  score_assistant_2 = None
                  explanation = f"Processing error: {e}"

                # Append results to the list
                evaluation_results.append({
                  "question": question,
                  "answer_asst_1":  answer_1,
                  "answer_asst_2":  answer_2,
                  "score_assistant_1": score_assistant_1,
                  "score_assistant_2": score_assistant_2,
                  "explanation": explanation
                })

            win_rate, counts = self.calculate_win_rate(evaluation_results)

            print(f"Total comparisons: {len(evaluation_results)}")
            print(f"Assistant 2 Wins: {counts['wins']}")
            print(f"Assistant 1 Wins: {counts['losses']}")
            print(f"Ties: {counts['ties']}")
            print(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")
            savefile ={}
            savefile["winrate"] = win_rate
            savefile["counts"] = counts
            savefile["results"] = evaluation_results
            # --- Output Results to JSON File ---
            try:
                with open(self.out_file_path, 'w', encoding='utf-8') as f:
                  json.dump(savefile, f, indent=4, ensure_ascii=False)
                print(f"\nEvaluation complete. Results saved to {self.out_file_path}")
            except Exception as e:
                print(f"An unexpected error occurred while saving results: {e}")

       
        else:
          print(f"Responses from Gemini-Judge already exist at {self.out_file_path}")        
          
          with open(self.out_file_path, "r") as f:
             evaluation_results = json.load(f)["results"]
          win_rate, counts = self.calculate_win_rate(evaluation_results)

          print(f"Total comparisons: {len(evaluation_results)}")
          print(f"Assistant 2 Wins: {counts['wins']}")
          print(f"Assistant 1 Wins: {counts['losses']}")
          print(f"Ties: {counts['ties']}")
          print(f"Win Rate for Assistant 2 (vs. Assistant 1): {win_rate}")