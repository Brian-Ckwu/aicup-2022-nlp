from typing import Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import Trainer

from model import BertEncoderNet

class BertEncoderNetTrainer(Trainer):
    
    def get_train_loader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            pin_memory=True
        )

    def get_eval_dataloader(self, eval_dataset = None):
        return DataLoader(
            dataset=self.eval_dataset,
            batch_size=self.args.eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            pin_memory=True
        )

    def log(self, logs: dict):
        # wandb.log
        print(logs)

    def compute_loss(self, model: BertEncoderNet, inputs: dict, return_outputs: bool = False): # inputs: {'X': BatchEncodings, 'y': (y_cls, y_seq)}
        logits_cls, logits_seq = model(inputs['X'])
        labels_cls, labels_seq = inputs['y']

        if model.multitask:
            loss = model.calc_mtl_loss(logits_cls, logits_seq, labels_cls, labels_seq)
        else:
            loss = model.calc_seq_loss(logits_seq, labels_seq)
        
        return (loss, (logits_cls, logits_seq)) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict, # {'X": BE, 'y': (y_cls, y_seq)}
        prediction_loss_only: bool,
        ignore_keys = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            labels_cls, labels_seq = inputs['y']
            loss, (logits_cls, logits_seq) = self.compute_loss(model, inputs, return_outputs=True)

        return (loss, None, None) if prediction_loss_only else (loss, logits_seq, labels_seq)

    # NOTE: currently only compute validation loss
    # TODO: add LCS calculation
    # def evaluate(self):
    #     raise NotImplementedError

    # def predict(self):
    #     raise NotImplementedError