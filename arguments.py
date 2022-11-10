from typing import List

import torch

from transformers import TrainingArguments
from dataclasses import dataclass

class Args(object):
    pass

class PreprocessArgs(Args):
    input_schemes = ["qr", "qrs"]
    output_schemes = ["q'r'", "sq'r'"]
    labeling_schemes = ["IO1", "IO2", "BIO1", "BIO2"] # 1: use the same class for q' and r' / 2: use different classes (e.g., I-q and I-r) for q' and r'
    
    def __init__(
        self,
        use_nltk: bool, # whether to use tokenize.word_tokenize() first before model_tokenizer
        model_tokenizer_name: str, # HuggingFace tokenizer name
        input_scheme: str, # qr | qrs
        output_scheme: str, # q'r' | sq'r'
        labeling_scheme: str, # IO1 | IO2 | BIO1 | BIO2
    ):
        self.use_nltk = use_nltk
        self.model_tokenizer_name = model_tokenizer_name
        self.set_input_scheme(input_scheme)
        self.set_output_scheme(output_scheme)
        self.set_labeling_scheme(labeling_scheme)

    def set_input_scheme(self, input_scheme: str) -> None:
        assert input_scheme in self.input_schemes
        self.input_scheme = input_scheme

    def set_output_scheme(self, output_scheme: str) -> None:
        assert output_scheme in self.output_schemes
        self.output_scheme = output_scheme

    def set_labeling_scheme(self, labeling_scheme: str) -> None:
        assert labeling_scheme in self.labeling_schemes
        self.labeling_scheme = labeling_scheme

class FileArgs(Args):

    def __init__(
        self,
        data_path: str,
        split_ids_path: str,
        save_dir: str = "./experiments"
    ):
        self.data_path = data_path
        self.split_ids_path = split_ids_path
        self.save_dir = save_dir
        # directory style:
        """
            - experiments
                - [exp_name]
                    - config.[yml/pickle]
                    - checkpoints
                        - [ckpt_name].pth
                        - ...
                    - eval_results
                        - [eval_name].[] # valid and test results
                        - ...
                    - inference_data
                        - [eval_name].[]
                        - ...
        """

class ModelArgs(Args):

    def __init__(
        self,
        model_type: str, # CNN / RNN / BERT / t5 / ...
        model_name: str, # HuggingFace model name
        w_loss_cls: float,
        w_loss_seq: float,
        checkpoint: str, # preivously trained checkpoint
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.w_loss_cls = w_loss_cls
        self.w_loss_seq = w_loss_seq
        self.checkpoint = checkpoint

class OptimizationArgs(Args):

    def __init__(
        self,
        lr: float, # learning rate
        nepochs: int,
        train_batch_size: int,
        eval_batch_size: int,
        grad_accum_steps: int,
        optimizer: str,
        scheduler: str,
        stopping_strategy: str
    ):
        self.lr = lr
        self.nepochs = nepochs
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.optimzier = optimizer
        self.scheduler = scheduler
        self.stoppping_strategy = stopping_strategy

@dataclass
class MyTrainingArguments(TrainingArguments):
    my_eval_metric_in_training: str = "token_acc"
    device_str: str = "cpu"

    @property
    def device(self) -> torch.device:
        return torch.device(self.device_str)
    
    @property
    def n_gpu(self) -> int:
        return 1

@dataclass
class WAndBArgs(object):
    project: str
    name: str
    tags: List[str]
    group: str

class ExperimentArgs(Args):

    def __init__(
        self,
        file_args: FileArgs,
        preprocess_args: PreprocessArgs,
        model_args: ModelArgs,
        train_args: MyTrainingArguments,
        wandb_args: WAndBArgs
    ):
        self.file_args = file_args
        self.preprocess_args = preprocess_args
        self.model_args = model_args
        self.train_args = train_args
        self.wandb_args = wandb_args
