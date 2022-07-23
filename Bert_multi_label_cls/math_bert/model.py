# -*- coding: utf-8 -*-
# __author__ = 'K_HOLMES_'
# __time__   = '2019/7/21 15:16'

#encoding:utf-8
import torch.nn as nn
from math_bert.novelty_modeling import BertPreTrainedModel, BertModel


class BertMath(BertPreTrainedModel):
    def __init__(self, bertConfig, num_classes):
        super(BertMath, self).__init__(bertConfig)
        self.bert = BertModel(bertConfig)  # bert模型
        self.dropout = nn.Dropout(bertConfig.hidden_dropout_prob)
        self.classifier = nn.Linear(in_features=bertConfig.hidden_size, out_features=num_classes)
        self.apply(self.init_bert_weights)
        self.unfreeze_bert_encoder()

    def freeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = False

    def unfreeze_bert_encoder(self):
        for p in self.bert.parameters():
            p.requires_grad = True

    def forward(self, input_ids, token_type_ids, attention_mask, input_ent=None, ent_mask=None,
                output_all_encoded_layers=False, get_cls=False):
        encoded_layers, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, input_ent, ent_mask,
                                                  output_all_encoded_layers=output_all_encoded_layers)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if not get_cls:
            return logits
        else:
            return pooled_output
