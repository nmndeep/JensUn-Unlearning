#!/bin/bash

conda activate LKFJensUn

export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
echo "Master Port: $MASTER_PORT"


model=Llama-3.2-3B-Instruct

evaltasks=(
   forget #evaluate the worst-case on the forget-set
   retain #evaluate the average-case on the retain-set
   mmlu #evaluate General ability on MMLLU
   repet #evaluate Repetitiveness on AlpacaEval
   winrate #evaluate WinRate c.f. base-model
)

runningargs=(
    
    "JensUn unlearn/lkf/paraphrases.yaml 10 "

)
for paramss in "${runningargs[@]}"; do
    trainer=$(echo $paramss | cut -d' ' -f1)
    experiment=$(echo $paramss | cut -d' ' -f2)
    epoch=$(echo $paramss | cut -d' ' -f3)

    echo "Epoch: $epoch"
    echo "LR: $lr"
   
    per_device_train_batch_size=4
    gradient_accumulation_steps=4
    echo "Batch size: $per_device_train_batch_size, Gradient Accumulation Steps: $gradient_accumulation_steps"

    task_name=LKF_${model}_${trainer}_epochs_${epoch}
    
    model_path=meta-llama/${model}

    echo ${task_name}: Unlearning ${model_path} using ${trainer}

    # Unlearn
    CUDA_VISIBLE_DEVICES=3,4 accelerate launch --config_file configs/accelerate/default_config.yaml --main_process_port $MASTER_PORT \
    src/train.py --config-name=unlearn.yaml \
    experiment=${experiment} \
    trainer=${trainer} \
    task_name=${task_name} \
    model=${model} \
    model.model_args.pretrained_model_name_or_path=${model_path} \
    retain_logs_path=saves/eval/tofu_${model}_${retain_split}/TOFU_EVAL.json \
    trainer.args.per_device_train_batch_size=$per_device_train_batch_size \
    trainer.args.gradient_accumulation_steps=$gradient_accumulation_steps \
    trainer.args.ddp_find_unused_parameters=true \
    trainer.args.gradient_checkpointing=true \
    trainer.args.num_train_epochs=$epoch \
    trainer.args.logging_steps=1 \
    trainer.args.eval_strategy=no \
    
    # Eval

    for task in "${evaltasks[@]}"; do
        if [[ $task == "retain" ]]; then
            ds_arg="dataset.name=nmndeep/LKF-retain_eval_para"
        else
            ds_arg=""
        fi
        CUDA_VISIBLE_DEVICES=8 python src/eval.py \
            experiment=eval.yaml \
            model.path=saves/unlearn/${task_name} \
            model.name=${model_path} \
            output.task_name=${task_name} \
            output.eval_task=${task} \
            ${ds_arg}
    
    done
done







