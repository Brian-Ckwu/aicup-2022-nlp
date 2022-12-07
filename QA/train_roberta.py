import json
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
from re import search
from typing import List, Tuple, Any, Union
import nltk
import torch
from datasets import Dataset
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoTokenizer
from tqdm.auto import tqdm
import collections
from sklearn.utils import shuffle

def load_json(file: str):
    return json.loads(Path(file).read_bytes())

p_data = pd.read_csv("./dataset/train.csv").drop(["Unnamed: 6", "total no.: 7987"], axis=1)
split_ids = load_json('./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json')
train_data, train2_data, valid_data = [p_data[p_data.id.isin(split_ids[split])] for split in ["train", "valid", "test"]]
print(train_data.shape[0],train2_data.shape[0],valid_data.shape[0])
train_data = pd.concat([train_data, train2_data],axis=0)
print(train_data.shape[0], valid_data.shape[0])
# train_data, valid_data, test_data = [p_data[p_data.id.isin(split_ids[split])] for split in ["train", "valid", "test"]]
# print(train_data.shape[0],valid_data.shape[0],test_data.shape[0])

# split_ids = load_json('./dataset/splitIds__nsplits-5_seed-3.json')
# train_data, valid_data = [p_data[p_data.id.isin(split_ids[3][split])] for split in ["train", "valid"]]
# print(p_data.shape[0], train_data.shape[0],valid_data.shape[0])
diff_seeds = [24, 16]
for diff_seed in diff_seeds:
    args = {
        "max_len" : 512,
        "batch_size" : diff_seed,
        "model_name" : "facebook/muppet-roberta-base",
        "learning_rate" : 3e-5,
        "warmup_ratio" : 0.06,
        "seed" : 24,
        "split" : "6+22",
        "do_train_shuffle": 1,
        "special": 'shuffle'
    }

    models_need_tokentypeid = ["google/electra-large-discriminator", "bert-based-uncased"]

    if args["do_train_shuffle"] == 1:
        train_data = shuffle(train_data)
        print(train_data[:10])

    tokenizer = AutoTokenizer.from_pretrained(args["model_name"])
    clsid = tokenizer.cls_token_id
    sepid = tokenizer.sep_token_id
    padid = tokenizer.pad_token_id

    def model_tokenize(text: str) -> List[int]:
        text = text.strip('"')
        token_ids = tokenizer(text)["input_ids"]
        return token_ids[1:-1] #without cls sep

    def contains(small, big):
        for i in range(len(big)-len(small)+1):
            for j in range(len(small)):
                if big[i+j] != small[j]:
                    break
            else:
                return i, i+len(small)-1
        return False

    def keep_continuous(data: pd.DataFrame):
        keep=[]
        for i in range(data.shape[0]):
            qp_not_in_q = data.iloc[i]['q\''][1:-1] not in data.iloc[i]['q'][1:-1]
            rp_not_in_r = data.iloc[i]['r\''][1:-1] not in data.iloc[i]['r'][1:-1]
            if not (qp_not_in_q or rp_not_in_r):
                keep.append(i)
        
        data = data.iloc[keep]
        return data

    #format data 
    #token type id sepid is in 0 not 1, context 0 question 1
    #attention mask有東西的1其他0
    #TODO: pad use tokenizer.pad_token_id
    def format_data_qp(q: List[int], r: List[int], s: int, qp: List[int], rp: List[int]) -> Tuple[List[int], List[int], List[int], int, int]:
        q_r_s = [clsid] + q + [sepid] + r + [sepid] + s + [sepid]
        attention_mask = [1 if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        input_id = [q_r_s[_] if _ in range(len(q_r_s)) else padid for _ in range(args['max_len'])]
        
        if contains(qp, q_r_s):
            start_pos, end_pos = contains(qp, q_r_s)
        else:
            start_pos, end_pos = 0, 0
            
        return input_id, attention_mask, start_pos, end_pos

    def format_data_rp(q: List[int], r: List[int], s: int, qp: List[int], rp: List[int]) -> Tuple[List[int], List[int], List[int], int, int]:
        q_r_s = [clsid] + r + [sepid] + q + [sepid] + s + [sepid]
        attention_mask = [1 if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        input_id = [q_r_s[_] if _ in range(len(q_r_s)) else padid for _ in range(args['max_len'])]
        
        if contains(rp, q_r_s):
            start_pos, end_pos = contains(rp, q_r_s)
        else:
            start_pos, end_pos = 0, 0
            
        return input_id, attention_mask, start_pos, end_pos

    def preprocess(data: pd.DataFrame, choice:str):
        data = keep_continuous(data)
        print('ids left:', data['id'].nunique())
        print('instances left', data.shape[0])
        ids = list(data.id)
        Q, R, S, QP, RP = [data[field] for field in ["q", "r", "s", "q'", "r'"]]
        Q, R, QP, RP, S = [list(map(model_tokenize, x)) for x in [Q, R, QP, RP, S]]
        
        # only keep those Q+R+S < 512 tokens
        count = 0
        keep = []
        for i in range(len(Q)):
            if (len(Q[i])+len(R[i])) > 512-7:
                count += 1
            else:
                keep.append(i)
        print(f"Q+R+S longer than {args['max_len']} tokens:", count, " Remains:",len(keep))
        Q = [Q[i] for i in keep]
        R = [R[i] for i in keep]
        QP = [QP[i] for i in keep]
        RP = [RP[i] for i in keep]
        S = [S[i] for i in keep]
        ids = [ids[i] for i in keep]
        
        #find start end positions make dict
        if choice == 'qp':
            data = list(map(format_data_qp, Q, R, S, QP, RP))
        elif choice == 'rp':
            data = list(map(format_data_rp, Q, R, S, QP, RP))
        else:
            return 'ERROR'
        input_list, token_list, attention_list, s_pos, e_pos =[], [], [], [], []
        for i in range(len(data)):
            input_list.append(data[i][0])
            attention_list.append(data[i][1])
            s_pos.append(data[i][2])
            e_pos.append(data[i][3])
            
        data = {
            'input_ids': input_list,
            'attention_masks': attention_list,
            'start_positions': s_pos,
            'end_positions': e_pos
        }
        
        #make dataset
        ds = Dataset.from_dict(data)
        return ds

    def format_data_qp_with_tokentypeid(q: List[int], r: List[int], s: int, qp: List[int], rp: List[int]) -> Tuple[List[int], List[int], List[int], int, int]:
        cls_q_sep = [clsid] + q + [sepid]
        token_type_id = [0 if _ in range(len(cls_q_sep)) else 1 for _ in range(args['max_len'])]
        
        q_r_s = [clsid] + q + [sepid] + r + [sepid] + s + [sepid]
        attention_mask = [1 if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        input_id = [q_r_s[_] if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        
        if contains(qp, q_r_s):
            start_pos, end_pos = contains(qp, q_r_s)
        else:
            start_pos, end_pos = 0, 0
            
        return input_id, token_type_id, attention_mask, start_pos, end_pos

    def format_data_rp_with_tokentypeid(q: List[int], r: List[int], s: int, qp: List[int], rp: List[int]) -> Tuple[List[int], List[int], List[int], int, int]:
        cls_q_sep = [clsid] + r + [sepid]
        token_type_id = [0 if _ in range(len(cls_q_sep)) else 1 for _ in range(args['max_len'])]
        
        q_r_s = [clsid] + r + [sepid] + q + [sepid] + s + [sepid]
        attention_mask = [1 if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        input_id = [q_r_s[_] if _ in range(len(q_r_s)) else 0 for _ in range(args['max_len'])]
        
        if contains(rp, q_r_s):
            start_pos, end_pos = contains(rp, q_r_s)
        else:
            start_pos, end_pos = 0, 0
            
        return input_id, token_type_id, attention_mask, start_pos, end_pos

    def preprocess_with_tokentypeid(data: pd.DataFrame, choice:str):
        data = keep_continuous(data)
        print('ids left:', data['id'].nunique())
        print('instances left', data.shape[0])
        ids = list(data.id)
        Q, R, S, QP, RP = [data[field] for field in ["q", "r", "s", "q'", "r'"]]
        Q, R, QP, RP, S = [list(map(model_tokenize, x)) for x in [Q, R, QP, RP, S]]
        
        # only keep those Q+R+S < 512 tokens
        count = 0
        keep = []
        for i in range(len(Q)):
            if (len(Q[i])+len(R[i])) > 512-5:
                count += 1
            else:
                keep.append(i)
        print(f"Q+R+S longer than {args['max_len']} tokens:", count, " Remains:",len(keep))
        Q = [Q[i] for i in keep]
        R = [R[i] for i in keep]
        QP = [QP[i] for i in keep]
        RP = [RP[i] for i in keep]
        S = [S[i] for i in keep]
        ids = [ids[i] for i in keep]
        
        #find start end positions make dict
        if choice == 'qp':
            data = list(map(format_data_qp_with_tokentypeid, Q, R, S, QP, RP))
        elif choice == 'rp':
            data = list(map(format_data_rp_with_tokentypeid, Q, R, S, QP, RP))
        else:
            return 'ERROR'
        input_list, token_list, attention_list, s_pos, e_pos =[], [], [], [], []
        for i in range(len(data)):
            input_list.append(data[i][0])
            token_list.append(data[i][1])
            attention_list.append(data[i][2])
            s_pos.append(data[i][3])
            e_pos.append(data[i][4])
            
        data = {
            'input_ids': input_list,
            'token_type_ids': token_list,
            'attention_masks': attention_list,
            'start_positions': s_pos,
            'end_positions': e_pos
        }
        
        #make dataset
        ds = Dataset.from_dict(data)
        return ds

    if args["model_name"] in models_need_tokentypeid:
        train_data_qp_done=preprocess_with_tokentypeid(train_data, 'qp')
        valid_data_qp_done=preprocess_with_tokentypeid(valid_data, 'qp')

        train_data_rp_done=preprocess_with_tokentypeid(train_data, 'rp')
        valid_data_rp_done=preprocess_with_tokentypeid(valid_data, 'rp')
    else:
        train_data_qp_done=preprocess(train_data, 'qp')
        valid_data_qp_done=preprocess(valid_data, 'qp')

        train_data_rp_done=preprocess(train_data, 'rp')
        valid_data_rp_done=preprocess(valid_data, 'rp')

    from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
    from transformers import default_data_collator

    model = AutoModelForQuestionAnswering.from_pretrained(args["model_name"])
    model_args = TrainingArguments(
        f'corrected_models/{args["model_name"]}-qp-{args["split"]}-b_{args["batch_size"]}-lr_{args["learning_rate"]}-warm_{args["warmup_ratio"]}-seed_{args["seed"]}-{args["special"]}',
        evaluation_strategy = "epoch",
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        save_steps=200,
        warmup_ratio=args["warmup_ratio"],
        seed=args["seed"]
    )


    data_collator = default_data_collator

    trainer = Trainer(
        model,
        model_args,
        train_dataset=train_data_qp_done,
        eval_dataset=valid_data_qp_done,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    #trainer.train(resume_from_checkpoint = True)
    #trainer.save_model(f'aicup-trained-qp-{args["model_name"]}-82')

    model = AutoModelForQuestionAnswering.from_pretrained(args["model_name"])
    model_args = TrainingArguments(
        f'corrected_models/{args["model_name"]}-rp-{args["split"]}-b_{args["batch_size"]}-lr_{args["learning_rate"]}-warm_{args["warmup_ratio"]}-seed_{args["seed"]}-{args["special"]}',
        evaluation_strategy = "epoch",
        learning_rate=args["learning_rate"],
        per_device_train_batch_size=args['batch_size'],
        per_device_eval_batch_size=args['batch_size'],
        num_train_epochs=5,
        weight_decay=0.01,
        gradient_accumulation_steps=2,
        save_steps=200,
        warmup_ratio=args["warmup_ratio"],
        seed=args["seed"]
    )


    data_collator = default_data_collator

    trainer = Trainer(
        model,
        model_args,
        train_dataset=train_data_rp_done,
        eval_dataset=valid_data_rp_done,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    #trainer.save_model(f'aicup-trained-rp-{args["model_name"]}-82')