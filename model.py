import torch
import torch.nn as nn

from transformers import AutoModel

class BertEncoderNet(nn.Module):

    def __init__(
        self, 
        model_name: str,
        num_tags: int # number of sequence labeling tags, depending on the labeling scheme
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.embed_dim = self.bert.embeddings.word_embeddings.embedding_dim
        self.fc_cls = nn.Linear(self.embed_dim, 2) # 0 == AGREE / 1 == DISAGREE
        self.fc_seq = nn.Linear(self.embed_dim, )

    def forward(self, x) -> tuple:
        h_last = self.bert(**x).last_hidden_state
        logits_cls = self.fc_cls(h_last[:, 0, :])
        logits_seq = self.fc_seq(h_last)
        return logits_cls, logits_seq