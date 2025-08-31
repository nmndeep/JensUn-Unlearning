import argparse
import json
import random
import os
from evals.judge_utils import *


class WorstEval:
    """Evaluates language model performance under "worst-case" scenarios, especially for forgetting tasks.

    This class reads evaluation results from two JSON files (typically representing
    evaluations with and without in-context retention,and paraphrased variations).
    It processes these results to determine "yes/no" answers for questions and then
    computes various accuracy metrics, including a "worst-case" accuracy,
    which likely combines performance across different evaluation settings.
    Results are logged to a JSONL file.

    Attributes:
        name (str): The name or identifier of the model being evaluated.
        out_dir (str): The base directory where the evaluation results will be saved.
        responsefiles (list[str]): A list of two file paths (strings) to JSON files
                                    containing model responses for evaluation.
        run_name (str): A unique identifier for the specific evaluation run.
        task (str): The type of task being evaluated (e.g., 'forget').
        logs (dict): A dictionary to store the final computed metrics for this evaluation.
        out_path (str): The full file path where the `self.logs` will be saved.
    """

    def __init__(self, name, run_name, worst_dir, files = None, task='forget'):

        self.name = name
        self.out_dir = worst_dir
        self.responsefiles = files
        self.run_name = run_name
        self.task = task
        
    def save_logs(self):
        """Save the logs in a json file"""
        os.makedirs(self.out_dir, exist_ok=True)
        modelname = self.name.split("/")[-1]
        filename = f"worst_case_eval_{modelname}_{self.run_name}_{self.task}.jsonl"
        self.out_path = os.path.join(self.out_dir, filename)
        with open(self.out_path, "w") as f:
            json.dump(self.logs, f, indent=4)

    def evaluate(self):
        """Performs the worst-case evaluation by processing model responses from files.

        Reads two sets of evaluation responses from `self.responsefiles`. For each set,
        it iterates through questions, converts responses to a 'yes/no' format
        using `check_yes_no`, and computes an overall average using `compute_any_one_average`.
        Finally, it computes a combined "worst-case" accuracy across both sets
        and stores all computed metrics in `self.out_path` before saving them.

        Prints:
            The 'J_P' (Judge_paraphrases), 'J_ICR' (Judge In-Context Retention),
            and 'J_W' (Judge Worst-Case) accuracy scores to the console.
        """
        with open(self.responsefiles[0]) as f:
            evals_para = json.load(f)
        with open(self.responsefiles[1]) as f:
            evals_icr = json.load(f)
        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())
        try:
            ans_newpara1 = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_para):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                # print(ans_list)
                ans_newpara1[f'{qi}'] = ans_list
            accs1, _ = compute_any_one_average(ans_newpara1, prefix="ans_")
        except:
            accs1=[None]
        
        print("J_P: ", accs1[-1])
        try: 
            ans_newpara2 = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_icr):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                ans_newpara2[f'{qi}'] = ans_list
            accs2, _ = compute_any_one_average(ans_newpara2, prefix="ans_")
        
        except:
            accs2=[None]
        print("J_ICR: ", accs2[-1])
        acc_tens = {}
        for qi in q_names:
            acc_tens[f'outs_1_{qi}'] = ans_newpara1[qi]
            acc_tens[f'outs_2_{qi}'] = ans_newpara2[qi]
        accov2, rowacc = compute_any_one_average(acc_tens)
        print("J_W: ", accov2[-1])
        self.logs = {
            "Model": self.name,
            "Task_name": self.run_name,
            "Set": self.task,
            "J_P": accs1[-1],
            "J_ICR": accs2[-1],
            "J_W": accov2[-1],
        }
        self.save_logs()


class AvgEval:
    """Evaluates language model performance for "average-case" scenarios, typically for retain set.

    This class reads evaluation results from a single JSON file, processes responses
    to 'yes/no' answers, and computes an "average-case" accuracy.
    Results are logged to a JSONL file, similar to `WorstEval` but focused on average performance.

    Attributes:
        name (str): The name or identifier of the model being evaluated.
        out_dir (str): The base directory where the evaluation results will be saved.
        responsefiles (list[str]): A list containing a single file path (string) to a JSON file
                                    containing model responses for evaluation.
        run_name (str): A unique identifier for the specific evaluation run.
        task (str): The type of task being evaluated (e.g., 'retain').
        logs (dict): A dictionary to store the final computed metrics for this evaluation.
        out_path (str): The full file path where the `self.logs` will be saved.
    """
    def __init__(self, name, run_name, worst_dir, files = None, task='retain'):

        self.name = name
        self.out_dir = worst_dir
        self.responsefiles = files
        self.run_name = run_name
        self.task = task
    
    def save_logs(self):
        """Save the logs in a json file"""
        os.makedirs(self.out_dir, exist_ok=True)
        modelname = self.name.split("/")[-1]
        filename = f"avg_case_eval_{modelname}_{self.run_name}_{self.task}.jsonl"
        self.out_path = os.path.join(self.out_dir, filename)
        with open(self.out_path, "w") as f:
            json.dump(self.logs, f, indent=4)

    def evaluate(self):
        """Performs the average-case evaluation by processing model responses from a file.

        Reads evaluation responses from the first file in `self.responsefiles`.
        It iterates through questions, converts responses to a 'yes/no' format
        using `check_yes_no`, and computes an overall average accuracy using
        `average_case_Acc`. Results are then stored in `self.out_path` and saved.

        Prints:
            The 'J_avg' (Judge Average) accuracy score to the console.
        """
        with open(self.responsefiles[0]) as f:
            evals_para = json.load(f)[:400]
        q_names = list({k: v for k, v in evals_para[0].items() if k != "GT"}.keys())
        try:
            ans_newpara1 = {}
            for qi in q_names:
                ans_list = []
                for ix, d in enumerate(evals_para):
                    ans = check_yes_no(d[f'{qi}'], 0)
                    ans_list.append(ans)
                # print(ans_list)
                ans_newpara1[f'{qi}'] = ans_list
            accs = average_case_acc(ans_newpara1, prefix="ans_")
        except:
            accs=None
        
        print("J_avg: ", accs)
     
        self.logs = {
            "Model": self.name,
            "Task_name": self.run_name,
            "Set": self.task,
            "J_avg": accs,
        }
        self.save_logs()


