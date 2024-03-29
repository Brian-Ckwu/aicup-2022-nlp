from typing import List, Tuple

import pandas as pd

import torch
from torch.utils.data import Dataset
from preprocess import Preprocessor

class BertEncoderNetDataset(Dataset):
    ignore_index = -100
    minimum_length_for_bigbird = 1024

    def __init__(self, r_data: pd.DataFrame, preprocessor: Preprocessor, do_eval: bool = False): # r_data: raw data
        self.r_data = r_data
        self.p_data = preprocessor(r_data.groupby("id").sample() if do_eval else r_data)
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.p_data)

    def __getitem__(self, idx: int):
        sample = self.p_data.iloc[idx].to_dict()
        X, (y_cls, y_seq) = sample['X'], sample['y']
        assert len(X) == len(y_seq)
        return X, (y_cls, y_seq)

    def collate_fn(self, samples: List[tuple]) -> Tuple[torch.Tensor]: # X, y_cls, y_seq
        X_l, y_cls_l, y_seq_l = list(), list(), list()
        for X, (y_cls, y_seq) in samples:
            X_l.append(X)
            y_cls_l.append(y_cls)
            y_seq_l.append(y_seq)

        # Pad X_l and y_seq_l
        X_l = self.pad_and_truncate(X_l, pad_id=self.preprocessor.model_tokenizer.pad_token_id, eos_id=self.preprocessor.model_tokenizer.sep_token_id)
        y_seq_l = self.pad_and_truncate(y_seq_l, pad_id=self.ignore_index)

        # Make token_type_ids masks & attention masks
        input_ids = torch.LongTensor(X_l)
        batch_X = {
            "input_ids": input_ids,
            "token_type_ids": self.make_token_type_ids(input_ids),
            "attention_mask": self.make_attention_mask(input_ids)
        }
        if self.preprocessor.model_tokenizer.name_or_path == "roberta-large":
            batch_X.pop("token_type_ids") # RoBERTa models have no token_type_ids

        batch_y_cls_l = torch.LongTensor(y_cls_l)
        batch_y_seq_l = torch.LongTensor(y_seq_l)
        return {
            'X': batch_X,
            'y': (batch_y_cls_l, batch_y_seq_l)
        }
    
    def pad_and_truncate(self, seq_l: List[List[int]], pad_id: int, eos_id: int = None) -> List[List[int]]:
        new_seq_l = list()
        if "bigbird" in self.preprocessor.model_tokenizer.name_or_path:
            max_len = min(self.preprocessor.model_tokenizer.model_max_length, max([len(seq) for seq in seq_l] + [self.minimum_length_for_bigbird]))
        else:
            max_len = min(self.preprocessor.model_tokenizer.model_max_length, max([len(seq) for seq in seq_l]))
        for seq in seq_l:
            if len(seq) > max_len: # truncate
                new_seq = seq[:max_len - 1] + [eos_id if eos_id is not None else pad_id]
            elif len(seq) < max_len: # pad
                new_seq = seq + [pad_id] * (max_len - len(seq))
            else:
                new_seq = seq
            new_seq_l.append(new_seq)

        return new_seq_l

    def make_token_type_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        token_type_ids = torch.zeros(size=input_ids.size()).long()
        segb_locs = (input_ids == self.preprocessor.model_tokenizer.sep_token_id).int().argmax(dim=1) + 1
        for i in range(len(segb_locs)):
            token_type_ids[i].index_fill_(dim=-1, index=torch.arange(segb_locs[i], token_type_ids.size(1)), value=1)
        return token_type_ids
    
    def make_attention_mask(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        attention_mask = torch.ones(size=input_ids.size()).long()
        padding_locs = (input_ids == self.preprocessor.model_tokenizer.pad_token_id).int()
        attention_mask += padding_locs * -1
        return attention_mask