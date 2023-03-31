from torch import nn
from transformers import BertModel, BertConfig


class BertClassifier(nn.Module):
    def __init__(self, num_classes, model_name, dropout):
        super().__init__()

        config = BertConfig(
            hidden_size=1024,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=1024
        )
        self.bert = BertModel.from_pretrained(model_name, config=config)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        return logits
