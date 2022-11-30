import json
from typing import Dict, List
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import EvalPrediction

from preprocess import Preprocessor
from postprocess import Postprocessor

def load_json(file: str):
    return json.loads(Path(file).read_bytes())

def compute_token_acc(logits_and_labels: EvalPrediction) -> Dict[str, float]:
    logits = logits_and_labels.predictions
    labels = logits_and_labels.label_ids

    preds = logits.argmax(axis=-1)
    actual_lens = (labels != -100).sum(axis=1)
    correct_lens = (preds == labels).sum(axis=1)
    acc = (correct_lens / actual_lens).mean().item()

    return {"token_acc": acc}

def longestCommonSubsequence(text1: list, text2: list) -> int:
    if len(text2) > len(text1):
        text1, text2 = text2, text1

    lcs = [[0] * (len(text2) + 1) for _ in range(2)]
    for i in range(1, len(text1)+1):
        for j in range(1, len(text2)+1):
            if text1[i-1] == text2[j-1]:
                lcs[i % 2][j] = lcs[(i-1) % 2][j-1] + 1
            else:
                lcs[i % 2][j] = max(lcs[(i-1) % 2][j], lcs[i % 2][j-1])

    return lcs[len(text1) % 2][len(text2)]

def compute_lcs_score(pred: list, ans: list) -> float:
    intersection = longestCommonSubsequence(pred, ans)
    union = len(pred) + len(ans) - intersection
    if union == 0:
        return 0
    lcs_score = intersection / union
    if (lcs_score < 0) or (lcs_score) > 1:
        raise ValueError("LCS score must be between 0 and 1")
    return lcs_score

def compute_lcs_scores(pred_df: pd.DataFrame, ans_df: pd.DataFrame) -> pd.DataFrame:
    ids, qp_scores, rp_scores = list(), list(), list()
    for _, prow in pred_df.iterrows():
        pid, qp_pred, rp_pred = prow["id"], prow["q'"], prow["r'"]
        qp_pred, rp_pred = [Preprocessor.nltk_tokenize(pred) for pred in [qp_pred, rp_pred]]
        ans_rows = ans_df[ans_df.id == pid]

        for _, arow in ans_rows.iterrows():
            qp_ans, rp_ans = arow["q'"], arow["r'"]
            qp_ans, rp_ans = [Preprocessor.nltk_tokenize(ans) for ans in [qp_ans, rp_ans]]
            qp_score, rp_score = compute_lcs_score(qp_pred, qp_ans), compute_lcs_score(rp_pred, rp_ans)

            for item, l in zip([pid, qp_score, rp_score], [ids, qp_scores, rp_scores]):
                l.append(item)

    assert ids == ans_df.id.tolist()
    lcs_df = pd.DataFrame(data={
        "id": ids,
        "qp_scores": qp_scores,
        "rp_scores": rp_scores
    })
    return lcs_df

def compute_final_score(lcs_df: pd.DataFrame) -> float:
    lcs_df["total_scores"] = lcs_df["qp_scores"] + lcs_df["rp_scores"]
    max_scores = lcs_df.groupby("id")["total_scores"].max()
    final_score = max_scores.sum() / (2 * len(max_scores))
    if (final_score < 0) or (final_score > 1):
        raise ValueError("The final score must be between 0 and 1, please check the implementation.")
    return final_score

def get_default_pred_df(r_data: pd.DataFrame):
    return r_data[["id", "q", "r"]].rename({"q": "q'", "r": "r'"}, axis=1).groupby("id").sample()

def get_interval2ids(p_data: pd.DataFrame) -> Dict[tuple, List[int]]:
    ntoken_intervals = [(0, 8)] + [(8 * 2 ** i, 8 * 2 ** (i + 1)) for i in range(9)] + [(4096, 10000)]
    ntokens = p_data.X.apply(len)
    interval2ids = {ntoken: list() for ntoken in ntoken_intervals}
    for interval in ntoken_intervals:
        low, high = interval
        indices = ntokens[(low < ntokens) & (ntokens <= high)].index
        ids = p_data.loc[indices].id.unique().tolist()
        interval2ids[interval] = ids

    return interval2ids

def get_max_lcs_df(lcs_df):
    lcs_df["total_scores"] = lcs_df["qp_scores"] + lcs_df["rp_scores"]
    max_scores = lcs_df.groupby("id")["total_scores"].max()
    return max_scores / 2

def calc_interval2score(pred_df: pd.DataFrame, ans_df: pd.DataFrame, interval2ids: Dict[tuple, List[int]]) -> Dict[tuple, float]:
    lcs_df = compute_lcs_scores(pred_df, ans_df)
    max_lcs_df = get_max_lcs_df(lcs_df)

    interval2score = dict()
    for interval, ids in interval2ids.items():
        score = max_lcs_df[max_lcs_df.index.isin(ids)].mean()
        interval2score[interval] = score

    return interval2score

def calc_default_interval2score(r_data: pd.DataFrame, p_data: pd.DataFrame) -> Dict[tuple, float]:
    pred_df = get_default_pred_df(r_data)
    ans_df = r_data[["id", "q'", "r'"]]
    interval2ids = get_interval2ids(p_data)
    interval2score = calc_interval2score(pred_df, ans_df, interval2ids)
    return interval2score

compute_metrics_funcs = {
    "token_acc": compute_token_acc
}

def ensemble_model_outputs(
    dataset, 
    trainer, 
    model: nn.Module, 
    model_ckpt_paths: List[str], 
    strategy: str, 
    device
):
    
    def sum_outputs(outputs_l: List[np.ndarray]) -> np.ndarray:
        summed = None
        for outputs in outputs_l:
            if summed is None:
                summed = outputs
            else:
                summed = summed + outputs
        return summed

    def sum_token_masks(logits_l: List[np.ndarray]) -> np.ndarray:
        summed = None
        for logits in logits_l:
            token_mask = logits.argmax(axis=-1)
            if summed is None:
                summed = token_mask
            else:
                summed = summed + token_mask
        return summed
    
    model_logits_l = list()
    for model_ckpt_path in model_ckpt_paths:
        print(f"----- Predicting outputs using checkpoint: {model_ckpt_path} -----")
        model.load_state_dict(torch.load(Path(model_ckpt_path) / "pytorch_model.bin", map_location=device))

        test_outputs = trainer.predict(dataset)
        logits = test_outputs.predictions
        model_logits_l.append(logits)

    if strategy == "logits":
        ensembled_logits = sum_outputs(model_logits_l)
        pred_token_mask_l = ensembled_logits.argmax(axis=-1)
    elif strategy == "softmax":
        model_softmax_l = [F.softmax(torch.tensor(model_logits), dim=-1) for model_logits in model_logits_l]
        ensembled_softmax = sum_outputs(model_softmax_l)
        pred_token_mask_l = ensembled_softmax.argmax(axis=-1)
    elif strategy == "voting":
        voting_threshold = len(model_ckpt_paths) // 2
        summed_token_mask = sum_token_masks(model_logits_l)
        pred_token_mask_l = torch.tensor(summed_token_mask > voting_threshold).int()
    else:
        raise ValueError("'strategy' must be 'logits', 'softmax', or 'voting'")
    
    return pred_token_mask_l

def calc_ensemble_score(dataset, pred_token_mask_l, return_lcs_df: bool = False) -> tuple:
    pred_sents = Postprocessor.predict_sents(dataset, pred_token_mask_l)

    ids = dataset.p_data.id.unique().tolist()

    pred_df = pd.DataFrame(data={
        "id": ids,
        **pred_sents
    })
    ans_df = dataset.r_data[["id", "q'", "r'"]]

    lcs_df = compute_lcs_scores(pred_df, ans_df)
    ensemble_score = compute_final_score(lcs_df)
    return ensemble_score if not return_lcs_df else (ensemble_score, lcs_df)

# for unit testing
if __name__ == "__main__":
    obj = load_json("./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json")
    print(obj)