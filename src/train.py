import hydra
from omegaconf import DictConfig
from trainer.utils import seed_everything

from data import get_collators, get_data
from model import get_model
from trainer import load_trainer

def change_at_runtime(eval_cfg:DictConfig, suffix:str):

    if isinstance(eval_cfg, DictConfig):
        for k, v in eval_cfg.items():
            # If there's an args section with question_key, update it
            if (
                isinstance(v, DictConfig)
                and "args" in v
                and isinstance(v.args, DictConfig)
                and "answer_key" in v.args
            ):
                v.args.answer_key = v.args.answer_key+suffix
                v.args.question_key = v.args.question_key+suffix # ugly fix
            else:
                # Recursively check nested configs
                change_at_runtime(v, suffix)

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    from omegaconf import OmegaConf
    import yaml
    with open('tmp-config-data.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(data_cfg), f)
    # change_at_runtime(data_cfg, '_real')
    with open('tmp-config-data-afterchange.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(data_cfg), f)
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")
    with open('trainerargs-cfg.yaml', 'w') as f:
        yaml.dump(OmegaConf.to_container(trainer_cfg), f)

    evaluator = None

    print(data.get("train", None))

    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )

    # save trainer args
    with open('trainerargs.yaml', 'w') as f:
        print(trainer_args)
        # print(type(trainer_args))
        yaml.dump(trainer_args.to_dict(), f)

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)
        print(f"FINAL_STRING:{trainer_args.output_dir}", flush=True)
        return trainer_args.output_dir

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")
        return trainer_args.output_dir

if __name__ == "__main__":
    otdir = main()
    print(f"FINAL_STRING:{otdir}", flush=True)
