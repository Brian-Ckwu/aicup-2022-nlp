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

    def run(self, debug_mode: bool): # debug_mode --> forward a small proportion of samples (e.g., 10 - 100)
        # Load data
        data = pd.read_csv(self.args.file_args.data_path).drop(["Unnamed: 6", "total no.: 7987"], axis=1)
        if debug_mode:
            data = data[:self.debug_nsamples]

        # Split into train/valid/test by ID
        train_data, valid_data, test_data = [data[data.id.isin(self.split_ids[split])] for split in ["train", "valid", "test"]]
        print(f"Sample size: train = {len(train_data)} / valid = {len(valid_data)} / test = {len(test_data)}")
        
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

        # TODO: Testing
            # post-processing
            # LCS

        self.wandb_run.summary["best_agg_lcs_score"] = trainer.state.best_metric
        self.wandb_run.finish()
        print(f"===== Best model checkpoint: {trainer.state.best_model_checkpoint} =====")
        (Path(self.args.train_args.output_dir) / "best_model_checkpoint.txt").write_text(trainer.state.best_model_checkpoint)

# Experiment Arguments
exp_name = "debug_run_for_save_ckpt"
exp_args = ExperimentArgs(
    file_args=FileArgs(
        data_path="./dataset/train.csv",
        split_ids_path="./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json"
    ),
    preprocess_args=PreprocessArgs(
        use_nltk=False,
        model_tokenizer_name="bert-base-uncased",
        input_scheme="qrs",
        output_scheme="q'r'",
        labeling_scheme="IO1"
    ),
    model_args=ModelArgs(
        model_type="bert",
        model_name="bert-base-uncased",
        w_loss_cls=None,
        w_loss_seq=None,
        checkpoint=None
    ),
    train_args=MyTrainingArguments(
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.0,
        seed=42,

        evaluation_strategy="epoch",
        # eval_steps=20,
        my_eval_metric_in_training="token_acc",
        logging_strategy="epoch",
        logging_first_step=True,
        # logging_steps=20,
        output_dir=f"./experiments/{exp_name}",
        overwrite_output_dir=True, # NOTE
        save_strategy="epoch",
        save_total_limit=5,
        # save_on_each_node=True,

        fp16=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_agg_lcs_score",
        greater_is_better=True,
        
        device_str="cuda:0",
    ),
    wandb_args=WAndBArgs(
        project="aicup",
        name=exp_name,
        tags=["debug", "baseline"],
        group="bert"
    ),
)

if __name__ == "__main__":
    exp = Experiment(exp_args)
    exp.run(debug_mode=True) # -> save experiment arguments and results to a specified directory