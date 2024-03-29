from typing import List, Tuple, Any, Union

import nltk

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from arguments import PreprocessArgs

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

class Preprocessor(object):
    punctuations = set([ch for ch in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"])
    s2id = {
        "AGREE": 0,
        "DISAGREE": 1
    }
    scheme2tags = {
        "IO1": {"O": 0, "I": 1},
        "IO2": {"O": 0, "I-q": 1, "I-r": 2},
        "BIO1": {"O": 0, "B": 1, "I": 2},
        "BIO2": {"O": 0, "B-q": 1, "I-q": 2, "B-r": 3, "I-r": 4}
    }
    ignore_index_symbol = 'X'
    ignore_index = -100
    default_model_max_length = 512

    def __init__(self, args: PreprocessArgs):
        self.args = args
        self.model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_name)

        if self.model_tokenizer.model_max_length > 1e8: # i.e., no model max length is specified
            self.model_tokenizer.model_max_length = self.default_model_max_length

    # TODO: how to design the default preprocessing function?
    def __call__(self, data: pd.DataFrame) -> pd.DataFrame: # converted DataFrame consists of X & y
        data.loc[:, ["q", "r", "s", "q'", "r'"]] = data[["q", "r", "s", "q'", "r'"]].applymap(lambda s: s.strip('"'))
        ids = data.id
        Q_raw, R_raw, S, QP_raw, RP_raw = [data[field] for field in ["q", "r", "s", "q'", "r'"]]
        # Tokenization of (q, r, q', r')
        if self.args.use_nltk:
            Q, R = [list(map(' '.join, map(self.nltk_tokenize, x))) for x in [Q_raw, R_raw]]
        Q, R = [list(map(self.model_tokenize, x)) for x in [Q_raw, R_raw]]
        Q, Q_offsets = zip(*Q)
        R, R_offsets = zip(*R)

        # Ground truth sequence labeling
        QP, RP = [list(map(self.label_sequence, ref, ans, ref_offsets)) for ref, ans, ref_offsets in [[Q_raw, QP_raw, Q_offsets], [R_raw, RP_raw, R_offsets]]]

        # Format the data
        X, X_offsets, y = zip(*list(map(self.format_data, Q, Q_offsets, R, R_offsets, S, QP, RP)))

        # Construct the preprocessed DataFrame
        pdf = pd.DataFrame(data={
            "id": ids,
            "X": X,
            "X_offsets": X_offsets,
            "y": y
        })
        return pdf

    def set_args(self, args: PreprocessArgs):
        self.args = args
    
    @classmethod # class method for convenience
    def nltk_tokenize(self, text: str, filter_puncts: bool = True) -> List[str]:
        text = text.strip('"') # NOTE: remove the quotes first
        tokens = nltk.tokenize.word_tokenize(text)
        if filter_puncts:
            tokens = list(filter(lambda t: t not in self.punctuations, tokens))
        return tokens

    def model_tokenize(self, text: str) -> List[int]:
        text = text.strip('"')
        be = self.model_tokenizer(text, return_offsets_mapping=True)
        token_ids = be["input_ids"]
        offsets = be["offset_mapping"]
        return token_ids[1:-1], offsets[1:-1] # NOTE: remove the first and the last token (e.g., the [CLS] token and the [SEP] token in BERT's tokenizer)
    
    # TODO: optimzie this function (problem: non-consecutive span, problematic sample: index == 11 of (q, q') pair)
    # def label_sequence_old(self, ref: List[Any], ans: List[Any]) -> List[int]:
    #     # check if LCS(ref, ans) = |ans|
    #     if longestCommonSubsequence(ref, ans) != len(ans):
    #         print(f"Reference sequence (len={len(ref)}): {ref}")
    #         print(f"Answer sequence (len={len(ans)}): {ans}")

    #     # label the reference sequence based on the answer sequence by two pointers
    #     i = 0
    #     j = 0
    #     labels = [0] * len(ref)
    #     while (j < len(ans)):
    #         if ref[i] == ans[j]:
    #             labels[i] = 1
    #             i += 1
    #             j += 1
    #         else:
    #             i += 1
        
    #     assert (np.array(ref)[np.array(labels).astype(bool)] == np.array(ans)).all()
    #     return labels

    @staticmethod
    def get_labeled_span_indices(ref: str, ans: str) -> List[List[int]]:
        s = 0
        e = len(ans)
        spans = list()
        while (s < len(ans)):
            cur_span = ans[s:e]
            span_s = ref.find(cur_span)
            if (span_s != -1):
                spans.append([span_s, span_s + len(cur_span)])
                s = e
                e = len(ans)
            else:
                e = e - 1
        return spans

    def label_sequence(self, ref: str, ans: str, ref_offsets: List[Tuple[int]]):
        labels = [0] * len(ref_offsets)
        labeled_spans = self.get_labeled_span_indices(ref, ans)

        if not labeled_spans:
            return labels

        cur = 0
        for i in range(len(ref_offsets)):
            cur_labeled_span = labeled_spans[cur]
            ref_offset = ref_offsets[i]
            if (cur_labeled_span[0] <= ref_offset[0]) and (ref_offset[1] <= cur_labeled_span[1]): # if the ref_offset is in the current labeled span
                labels[i] = 1
            elif (ref_offset[0] < cur_labeled_span[0]): # if the ref_offset is to the left of current labeled span
                pass # do nothing
            elif (ref_offset[1] > cur_labeled_span[1]):
                cur += 1
                if cur >= len(labeled_spans):
                    break
            else:
                raise Exception("This condition should be happen.")

        return labels

    def format_data(
        self, 
        q: List[int], 
        q_offsets: List[Tuple[int]],
        r: List[int], 
        r_offsets: List[Tuple[int]],
        s: str, 
        qp: List[int], 
        rp: List[int]
    ) -> Tuple[List[int], Union[List[int], Tuple[int, List[int]]]]:
        assert (len(q) == len(qp)) and (len(r) == len(rp))
        
        # TODO: adapt to other model type
        clsid = self.model_tokenizer.cls_token_id
        sepid = self.model_tokenizer.sep_token_id
        nofst = [(0, 0)]
        
        qr = [clsid] + q + [sepid] + r + [sepid]
        qoro = nofst + q_offsets + nofst + r_offsets
        # handle labeling scheme
        qprp = getattr(self, f"to_{self.args.labeling_scheme}_scheme")(qp, rp)

        # handle input scheme
        X = qr
        X_offsets = qoro
        if self.args.input_scheme == "qrs": # append s -> [CLS] q [SEP] r [SEP] s [SEP]
            X += [self.model_tokenizer.convert_tokens_to_ids(s.lower()), sepid] # lowercased
            X_offsets += nofst * 2
            qprp += [self.ignore_index_symbol] * 2

        # handle output scheme
        y_cls = self.ignore_index # unify the return values
        y_seq = self.to_int_seq(qprp) # convert BIO tagging to integer labels
        if self.args.output_scheme == "sq'r'":
            y_cls = self.s2id[s]
        return X, X_offsets, (y_cls, y_seq)

    def to_IO1_scheme(self, qp: List[int], rp: List[int]) -> List[str]: # "IO1": {"O": 0, "I": 1}
        nqp = ['O'] * len(qp)
        nrp = ['O'] * len(rp)

        for new_seq, ori_seq in [[nqp, qp], [nrp, rp]]:
            for i in range(len(ori_seq)):
                if ori_seq[i] == 1:
                    new_seq[i] = 'I'

        return self.add_ignores(nqp, nrp)

    def to_IO2_scheme(self, qp: List[int], rp: List[int]) -> List[str]: # O, I-q, I-r
        nqp = ['O'] * len(qp)
        nrp = ['O'] * len(rp)

        for seq_id, (new_seq, ori_seq) in enumerate([[nqp, qp], [nrp, rp]]):
            label = "I-q" if seq_id == 0 else "I-r"
            for i in range(len(ori_seq)):
                if ori_seq[i] == 1:
                    new_seq[i] = label                

        return self.add_ignores(nqp, nrp)

    def to_BIO1_scheme(self, qp: List[int], rp: List[int]) -> List[str]: # O, B, I
        nqp = ['O'] * len(qp)
        nrp = ['O'] * len(rp)

        for new_seq, ori_seq in [[nqp, qp], [nrp, rp]]:
            in_span = False
            for i in range(len(ori_seq)):
                if in_span:
                    if ori_seq[i] == 0:
                        in_span = False
                    else: # ori_seq[i] == 1
                        new_seq[i] = 'I'
                else: # previously not in span
                    if ori_seq[i] == 1:
                        new_seq[i] = 'B'
                        in_span = True

        return self.add_ignores(nqp, nrp)

    def to_BIO2_scheme(self, qp: List[int], rp: List[int]) -> List[str]: # O, B-q, I-q, B-r, I-r
        nqp = ['O'] * len(qp)
        nrp = ['O'] * len(rp)        
        
        for seq_id, (new_seq, ori_seq) in enumerate([[nqp, qp], [nrp, rp]]):
            seq_symbol = 'q' if seq_id == 0 else 'r'
            B = f"B-{seq_symbol}"
            I = f"I-{seq_symbol}"
            in_span = False
            for i in range(len(ori_seq)):
                if in_span:
                    if ori_seq[i] == 0:
                        in_span = False
                    else: # ori_seq[i] == 1
                        new_seq[i] = I
                else: # previously not in span
                    if ori_seq[i] == 1:
                        new_seq[i] = B
                        in_span = True

        return self.add_ignores(nqp, nrp)                    

    def to_int_seq(self, seq: List[str]) -> List[int]:
        nseq = [0] * len(seq) # default to a sequence of ['O']
        for i in range(len(seq)):
            sym = seq[i]
            if sym == self.ignore_index_symbol:
                nseq[i] = self.ignore_index
            else:
                nseq[i] = self.scheme2tags[self.args.labeling_scheme][sym]

        return nseq

    @staticmethod
    def add_ignores(seq1: List[str], seq2: List[str]) -> List[str]:
        return ['X'] + seq1 + ['X'] + seq2 + ['X']

    # Deprecated functions
    @staticmethod
    def add_labels_by_two(seq: List[int]) -> List[int]:
        new_seq = seq.copy()
        for i in range(len(new_seq)):
            if new_seq[i] != 0: # i.e., annotated span
                new_seq[i] += 2
        return new_seq

    @staticmethod
    def add_B_to_labels(seq: List[int]) -> List[int]:
        new_seq = seq.copy()
        in_span = False # flag to keep track of whether the pointer is in the annotated span
        for i in range(len(new_seq)):
            if (in_span) and (new_seq[i] % 2 == 0):
                in_span = False
            elif (not in_span) and (new_seq[i] % 2 == 1): # the beginning of the annotated span
                new_seq[i] += 1
                in_span = True
        return new_seq
