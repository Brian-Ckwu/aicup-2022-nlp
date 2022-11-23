import json
from typing import Dict, List
from pathlib import Path

import pandas as pd

from transformers import EvalPrediction

from preprocess import Preprocessor

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

# for unit testing
if __name__ == "__main__":
    obj = load_json("./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json")
    print(obj)