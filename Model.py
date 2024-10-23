# model.py
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification, RobertaForSequenceClassification, BertModel
from Config import DEVICE


# 扩展模型，加入情感特征
class SentimentClassifierWithEmotion(nn.Module):
    def __init__(self, num_labels=6):
        super(SentimentClassifierWithEmotion, self).__init__()
        # 使用预训练的BERT模型
        self.bert = BertModel.from_pretrained('bert-base-chinese')

        # 定义Dropout层防止过拟合
        self.dropout = nn.Dropout(0.3)

        # 分类器：BERT的输出 + 情感特征
        self.classifier = nn.Linear(self.bert.config.hidden_size + 1, num_labels)  # +1是因为情感特征是单一数值

    def forward(self, input_ids, attention_mask, emotion_features):
        # 获取 BERT 的输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # BERT 的 [CLS] token 的输出

        # 拼接 BERT 的输出和情感特征
        combined_output = torch.cat((pooled_output, emotion_features.float()), dim=1)

        # 通过 Dropout 层和分类层
        logits = self.classifier(self.dropout(combined_output))

        return logits


# 在 build_model 中构建这个扩展的模型
def build_model(num_labels=6):
    # 使用扩展的情感特征分类模型
    model = SentimentClassifierWithEmotion(num_labels=num_labels)
    model.to(DEVICE)
    return model