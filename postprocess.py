from typing import List, Tuple

import torch

class Postprocessor(object):

    def __init__(self, model_tokenizer):
        self.model_tokenizer = model_tokenizer

    @staticmethod
    def aggregate_dataset_inputs(dataset):
        samples = list()
        for i in range(len(dataset)):
            sample = dataset[i]
            samples.append(sample)

        collated = dataset.collate_fn(samples)
        return collated['X']["input_ids"]

    @staticmethod
    def merge_offsets(offsets: List[List[int]], mask: List[int], window: Tuple[int]):
        s, e = window # start, end
        w_offsets = offsets[s:e]
        w_mask = mask[s:e]

        m_offsets = list() # merged offsets
        prev_in_span = False
        span = None
        for w_offset, in_span in zip(w_offsets, w_mask):
            if prev_in_span:
                if in_span:
                    span[1] = w_offset[1]
                else:
                    m_offsets.append(span)
                    span = None
                    prev_in_span = False
            else:
                if in_span:
                    span = list(w_offset)
                    prev_in_span = True
        
        if span: # push the last span
            m_offsets.append(span)
        return m_offsets

    def return_output_sents(self, raw_eval_data, eval_token_ids_l, offsets_l, pred_token_mask_l):
        assert len(raw_eval_data) == len(eval_token_ids_l) == len(offsets_l) == len(pred_token_mask_l)
        qp_sents, rp_sents = list(), list()

        for i in range(len(raw_eval_data)):
            raw_q = raw_eval_data.iloc[i].q.strip('"')
            raw_r = raw_eval_data.iloc[i].r.strip('"')

            token_ids = eval_token_ids_l[i]
            offsets = offsets_l.iloc[i]
            mask = pred_token_mask_l[i]

            sep_locs = torch.nonzero(token_ids == self.model_tokenizer.sep_token_id).flatten()
            qp_offsets = self.merge_offsets(offsets, mask, window=(1, sep_locs[0]))
            if len(sep_locs) >= 2:
                rp_offsets = self.merge_offsets(offsets, mask, window=(sep_locs[0] + 1, sep_locs[1]))
            else:
                rp_offsets = torch.tensor([])

            qp_sent = ' '.join([raw_q[s:e] for s, e in qp_offsets])
            rp_sent = ' '.join([raw_r[s:e] for s, e in rp_offsets])

            qp_sents.append('"' + qp_sent + '"')
            rp_sents.append('"' + rp_sent + '"')
        
        pred_sents = {
            'q': qp_sents,
            'r': rp_sents
        }
        return pred_sents