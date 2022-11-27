"""
    Import the packages
"""
import os
import wandb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import itertools
from pathlib import Path

import pandas as pd
from transformers import logging
logging.set_verbosity_error()

from utils import load_json, compute_metrics_funcs
from model import BertEncoderNet
from data import BertEncoderNetDataset
from trainer import BertEncoderNetTrainer
from arguments import PreprocessArgs, FileArgs, ModelArgs, MyTrainingArguments, WAndBArgs, ExperimentArgs
from preprocess import Preprocessor

class Experiment(object):
    debug_nsamples = 500

    def __init__(self, args: ExperimentArgs):
        print("Initializing the objects...")
        self.args = args
        self.split_ids = load_json(file=args.file_args.split_ids_path)
        self.preprocessor = Preprocessor(args=args.preprocess_args)
        self.model = BertEncoderNet(
            model_name=args.model_args.model_name, 
            num_tags=len(Preprocessor.scheme2tags[args.preprocess_args.labeling_scheme]),
            multitask=(args.preprocess_args.output_scheme == "sq'r'"),
            w_loss_cls=args.model_args.w_loss_cls,
            w_loss_seq=args.model_args.w_loss_seq
        )
        self.wandb_run = wandb.init(reinit=True, **vars(args.wandb_args))
        if self.args.cross_validation_idx is not None:
            self.split_ids = self.split_ids[self.args.cross_validation_idx]

    def run(self, debug_mode: bool): # debug_mode --> forward a small proportion of samples (e.g., 10 - 100)
        # Load data
        data = pd.read_csv(self.args.file_args.data_path).drop(["Unnamed: 6", "total no.: 7987"], axis=1)
        if debug_mode:
            data = data[:self.debug_nsamples]

        # Split into train/valid/test by ID
        train_data, valid_data = [data[data.id.isin(self.split_ids[split])] for split in ["train", "valid"]]
        print(f"Sample size: train = {len(train_data)} / valid = {len(valid_data)}")
        
        # Make datasets
        print("Building datasets...")
        train_dataset = BertEncoderNetDataset(r_data=train_data, preprocessor=self.preprocessor)
        valid_dataset = BertEncoderNetDataset(r_data=valid_data, preprocessor=self.preprocessor, do_eval=True) # do_eval -> group IDs
        print(f"Dataset size: train = {len(train_dataset)} / valid = {len(valid_dataset)}")

        # TODO: Load model (if resume from a previously trained checkpoint)
        if self.args.model_args.checkpoint:
            raise NotImplementedError

        # Optimization (Trainer?): training + validation # TODO: log to wandb
        trainer = BertEncoderNetTrainer(
            model=self.model,
            args=self.args.train_args,
            data_collator=train_dataset.collate_fn,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            compute_metrics=compute_metrics_funcs[self.args.train_args.my_eval_metric_in_training]
        )
        trainer.evaluate() # evaluate before any training
        trainer.train()

        self.wandb_run.summary["best_agg_lcs_score"] = trainer.state.best_metric
        self.wandb_run.finish()
        print(f"===== Best model checkpoint: {trainer.state.best_model_checkpoint} =====")
        (Path(self.args.train_args.output_dir) / "best_model_checkpoint.txt").write_text(trainer.state.best_model_checkpoint)

if __name__ == "__main__":
    # Variables
    mode = "submission"
    model_size = "large"
    huggingface_model = f"roberta-{model_size}"
    steps = 500
    lr = 2e-5
    warmup_ratio = 0.06
    seeds = range(5, 10)
    input_scheme = "qrs"
    device = "cuda:1"

    # Automatically assigned constants
    cv_mode = "submission" if mode == "submission" else "experiments"
    cv_type = "final_submission" if mode == "submission" else "our_testing"
    strategy = "epoch" if mode == "debug" else "steps"
    n_splits = 5

    for seed in seeds:
        for cv_idx in range(n_splits):
            print(f"\nCurrently cross validation @seed-{seed}/split-{cv_idx}:")
            exp_name = f"{mode}_run_{huggingface_model}_cvseed-{seed}_split-{cv_idx}_lr-{lr}_warmup-{int(warmup_ratio * 100)}pct"
            exp_args = ExperimentArgs(
                file_args=FileArgs(
                    data_path="./dataset/train.csv",
                    split_ids_path=f"./dataset/cross_validation/{cv_type}/splitIds__nsplits-5_seed-{seed}.json"
                ),
                preprocess_args=PreprocessArgs(
                    use_nltk=False,
                    model_tokenizer_name=huggingface_model,
                    input_scheme=input_scheme,
                    output_scheme="q'r'",
                    labeling_scheme="IO1"
                ),
                model_args=ModelArgs(
                    model_type="bert",
                    model_name=huggingface_model,
                    w_loss_cls=None,
                    w_loss_seq=None,
                    checkpoint=None
                ),
                train_args=MyTrainingArguments(
                    num_train_epochs=4,
                    per_device_train_batch_size=4,
                    per_device_eval_batch_size=8,
                    gradient_accumulation_steps=4,
                    learning_rate=lr,
                    lr_scheduler_type="linear",
                    warmup_ratio=warmup_ratio,
                    seed=seed,

                    evaluation_strategy=strategy,
                    eval_steps=steps,
                    my_eval_metric_in_training="token_acc",
                    logging_strategy=strategy,
                    logging_first_step=True,
                    logging_steps=steps,
                    output_dir=f"./{cv_mode}/cross-validation/{huggingface_model}/cvseed-{seed}_idx-{cv_idx}_lr-{lr}_warmup-{int(warmup_ratio * 100)}pct",
                    overwrite_output_dir=True, # NOTE
                    save_strategy=strategy,
                    save_total_limit=2,

                    fp16=False,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_agg_lcs_score",
                    greater_is_better=True,
                    
                    device_str=device,
                ),
                wandb_args=WAndBArgs(
                    project="aicup",
                    name=exp_name,
                    tags=[mode, huggingface_model],
                    group=f"{mode}-cv-seed-{seed}_lr-{lr}_warmup-{int(warmup_ratio * 100)}pct"
                ),
                cross_validation_idx=cv_idx
            )

            exp = Experiment(exp_args)
            exp.run(debug_mode=(mode == "debug")) # -> save experiment arguments and results to a specified directory