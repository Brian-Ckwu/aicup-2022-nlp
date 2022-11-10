import json
from typing import Dict
from pathlib import Path

from transformers import EvalPrediction

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

compute_metrics_funcs = {
    "token_acc": compute_token_acc
}

# for unit testing
if __name__ == "__main__":
    obj = load_json("./dataset/splitIds__splitBy-id_stratifyBy-s_train-0.6_valid-0.2_test-0.2_seed-42.json")
    print(obj)