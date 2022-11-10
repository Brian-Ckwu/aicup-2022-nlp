import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel

class BertEncoderNet(nn.Module):

    def __init__(
        self, 
        model_name: str,
        num_tags: int, # number of sequence labeling tags, depending on the labeling scheme
        multitask: bool,
        w_loss_cls: float = None,
        w_loss_seq: float = None,
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.fc_cls = nn.Linear(self.embed_dim, 2) # 0 == AGREE / 1 == DISAGREE
        self.fc_seq = nn.Linear(self.embed_dim, num_tags)

        self.multitask = multitask
        if multitask:
            assert (type(w_loss_cls) == float) and (type(w_loss_seq) == float)
        self.w_loss_cls = w_loss_cls
        self.w_loss_seq = w_loss_seq

    def forward(self, x) -> tuple:
        h_last = self.bert(**x).last_hidden_state
        logits_cls = self.fc_cls(h_last[:, 0, :])
        logits_seq = self.fc_seq(h_last)
        return logits_cls, logits_seq

    def calc_cls_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        return F.cross_entropy(input=logits, target=labels)

    def calc_seq_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        return F.cross_entropy(input=logits.transpose(1, 2), target=labels)

    def calc_mtl_loss(
        self, 
        logits_cls: torch.Tensor, 
        logits_seq: torch.Tensor, 
        labels_cls: torch.LongTensor,
        labels_seq: torch.LongTensor
    ) -> torch.Tensor:
        assert self.multitask
        loss_cls = self.calc_cls_loss(logits_cls, labels_cls)
        loss_seq = self.calc_seq_loss(logits_seq, labels_seq)
        loss_mtl = self.w_loss_cls * loss_cls + self.w_loss_seq * loss_seq
        return loss_mtl