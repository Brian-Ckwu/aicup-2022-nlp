import math
import time
from typing import Tuple, Optional, List, Dict

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.debug_utils import DebugOption

from transformers.file_utils import is_torch_tpu_available

from data import BertEncoderNetDataset
from model import BertEncoderNet
from utils import compute_lcs_scores, compute_final_score
from postprocess import Postprocessor

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

    # def log(self, logs: dict):
    #     # wandb.log
    #     print(logs)

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
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is an `datasets.Dataset`,
                columns not accepted by the `model.forward()` method are automatically removed. It must implement the
                `__len__` method.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # Compute lcs score and add to metrics
        pred_outputs = self.predict(eval_dataloader.dataset)
        pred_token_mask_l = pred_outputs.predictions.argmax(axis=-1)
        pred_sents = Postprocessor.predict_sents(dataset=eval_dataloader.dataset, pred_token_mask_l=pred_token_mask_l)

        pred_df = pd.DataFrame(data={
            "id": eval_dataloader.dataset.p_data.id.unique().tolist(),
            **pred_sents
        })
        ans_df = eval_dataloader.dataset.r_data[["id", "q'", "r'"]]

        agg_lcs_score = compute_final_score(compute_lcs_scores(pred_df, ans_df))
        output.metrics["eval_agg_lcs_score"] = agg_lcs_score

        self.log(output.metrics)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics