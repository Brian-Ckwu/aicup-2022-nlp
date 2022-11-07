from typing import List, Tuple, Any, Union

import nltk

import numpy as np
from transformers import AutoTokenizer

from utils import longestCommonSubsequence
from arguments import PreprocessArgs

class Preprocessor(object):
    punctuations = set([ch for ch in "!\"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"])
    s2id = {
        "AGREE": 0,
        "DISAGREE": 1
    }
    ignore_index = -100

    def __init__(self, args: PreprocessArgs):
        self.args = args
        self.model_tokenizer = AutoTokenizer.from_pretrained(args.model_tokenizer_name)

    def set_args(self, args: PreprocessArgs):
        self.args = args
    
    def nltk_tokenize(self, text: str, filter_puncts: bool = True) -> List[str]:
        tokens = nltk.tokenize.word_tokenize(text)
        if filter_puncts:
            tokens = list(filter(lambda t: t not in self.punctuations, tokens))
        return tokens

    def model_tokenize(self, text: str) -> List[int]:
        text = text.strip('"')
        token_ids = self.model_tokenizer(text)["input_ids"]
        return token_ids[1:-1] # NOTE: remove the first and the last token (e.g., the [CLS] token and the [SEP] token in BERT's tokenizer)
    
    # TODO: optimzie this function (problem: non-consecutive span, problematic sample: index == 11 of (q, q') pair)
    def label_sequence(self, ref: List[Any], ans: List[Any]) -> List[int]:
        # check if LCS(ref, ans) = |ans|
        assert longestCommonSubsequence(ref, ans) == len(ans)

        # label the reference sequence based on the answer sequence by two pointers
        i = 0
        j = 0
        labels = [0] * len(ref)
        while (j < len(ans)):
            if ref[i] == ans[j]:
                labels[i] = 1
                i += 1
                j += 1
            else:
                i += 1
        
        assert (np.array(ref)[np.array(labels).astype(bool)] == np.array(ans)).all()
        return labels

    def format_data(self, q: List[int], r: List[int], s: str, qp: List[int], rp: List[int]) -> Tuple[List[int], Union[List[int], Tuple[int, List[int]]]]:
        assert (len(q) == len(qp)) and (len(r) == len(rp))
        
        # TODO: adapt to other model type
        clsid = self.model_tokenizer.cls_token_id
        sepid = self.model_tokenizer.sep_token_id
        
        qr = [clsid] + q + [sepid] + r + [sepid]
        # handle labeling scheme - 1/2
        if self.args.labeling_scheme[-1] == '2': # use separate labels for the second sequence
            rp = self.add_labels_by_two(rp)
        qprp = [self.ignore_index] + qp + [self.ignore_index] + rp + [self.ignore_index]

        # handle input scheme
        X = qr
        if self.args.input_scheme == "qrs": # append s -> [CLS] q [SEP] r [SEP] s [SEP]
            X += [self.model_tokenizer.convert_tokens_to_ids(s.lower()), sepid] # lowercased
            qprp += [self.ignore_index, self.ignore_index]

        # handle labeling scheme - IO/BIO
        y_seq = qprp
        if self.args.labeling_scheme[:-1] == "BIO":
            y_seq = self.add_B_to_labels(y_seq)

        # handle output scheme
        if self.args.output_scheme == "sq'r'":
            y_cls = self.s2id[s]
            return X, (y_cls, y_seq)
        return X, y_seq

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
