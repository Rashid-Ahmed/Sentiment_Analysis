from torch import nn
from transformers import BertModel, BertConfig


#LSTM model we use for sentiment analysis
class BertClassifier(nn.Module):
    def __init__(self, num_classes, model_name, dropout):
        
        super(BertClassifier, self).__init__()
        
        configuration = BertConfig(hidden_size = 1024, num_hidden_layers = 4, num_attention_heads = 4, intermediate_size = 1024)
        self.bert = BertModel.from_pretrained(model_name)
        self.bert = BertModel(configuration)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_id, mask):
        
        
        _, bert_output = self.bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        output = self.dropout(bert_output)
        output = self.linear(output)
        
        return output
		    


    