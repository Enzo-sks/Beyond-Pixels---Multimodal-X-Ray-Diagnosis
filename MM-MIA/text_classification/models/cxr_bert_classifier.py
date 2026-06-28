"""cxr_bert_classifier.py — CXR-BERT-specialized + multi-label head."""

import torch.nn as nn
from transformers import AutoModel

CXR_BERT_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"


class CXRBertClassifier(nn.Module):
    def __init__(self, n_labels, pad_id=0, model_name=CXR_BERT_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, n_labels))
        self.pad_id = pad_id

        core = self.bert.bert if hasattr(self.bert, "bert") else self.bert
        self.encoder_layers = core.encoder.layer
        self.pooler = getattr(core, "pooler", None)

    def forward(self, ids):
        mask = (ids != self.pad_id).long()
        out = self.bert(input_ids=ids, attention_mask=mask)
        return self.head(out.last_hidden_state[:, 0])

    def freeze_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self):
        """Unfreeze layers 10-11 + pooler."""
        for layer in self.encoder_layers[10:]:
            for p in layer.parameters():
                p.requires_grad = True
        if self.pooler is not None:
            for p in self.pooler.parameters():
                p.requires_grad = True
