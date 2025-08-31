#!/bin/bash

# conda activate LKFJensUn

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"

model="Llama-3.2-3B-Instruct"


evaltasks=(
   forget
   retain
   mmlu
   repet 
   winrate
)

#Change based on the unlearning-method, similar to the one in lkf_unlearn.sh
task_name=pretrained

for task in "${evaltasks[@]}"; do
    if [[ $task == "retain" ]]; then
        ds_arg="dataset.name=nmndeep/LKF-retain_eval_para"
    else
        ds_arg=""
    fi
    CUDA_VISIBLE_DEVICES=0 python src/eval.py \
        experiment=eval.yaml \
        model.path=meta-llama/${model} \
        model.name=meta-llama/${model} \
        output.task_name=${task_name} \
        output.eval_task=${task} \
        ${ds_arg}

done    
