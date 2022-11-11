import pandas as pd

from pathlib import Path
from argparse import Namespace, ArgumentParser

from utils import compute_lcs_scores, compute_final_score

def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument(
        "--prediction",
        type=Path,
        help="Path to the prediction file (csv).",
        required=True
    )
    parser.add_argument(
        "--answer",
        type=Path,
        help="Path to the answer file (csv).",
        required=True
    )

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    pred_df = pd.read_csv(args.prediction, names=["id", "q'", "r'"])
    ans_df = pd.read_csv(args.answer, names=["id", "q'", "r'"])
    if len(pred_df) != len(ans_df.groupby("id").size()):
        raise ValueError("The prediction file must have the same number of rows as the number of unique IDs in the answer file")
    
    lcs_df = compute_lcs_scores(pred_df, ans_df) # has len(ans_df) rows of lcs_q' and lcs_r'
    final_score = compute_final_score(lcs_df) # derive the final score by "1/2N (\sum_i^N(max_j(score_q' + score_r')))"
    print(f"Final test score: {final_score}")

    # TODO: save the final_score to corresponding model checkpoint's folder