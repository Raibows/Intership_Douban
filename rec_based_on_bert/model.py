import torch
from transformers import BertModel
from torch import nn
from config import model_config



class bert_douban(nn.Module):
    def __init__(self, bert_dir_path, hidden_size, num_layer, label_num, dropout, require_grad):
        super().__init__()
        assert num_layer > 1
        self.bert = BertModel.from_pretrained(bert_dir_path)
        for param in self.bert.parameters():
            param.requires_grad = require_grad
        layers = [nn.Dropout(dropout), nn.Linear(768, hidden_size), nn.Dropout(dropout), nn.ReLU()]
        for i in range(num_layer-2):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, label_num))
        self.fc = nn.Sequential(*layers)


    def forward(self, x):
        encoders, pooled = self.bert(input_ids=x[0],
                                     token_type_ids=x[1],
                                     attention_mask=x[2],)[:]
        logits = self.fc(pooled)

        return logits


if __name__ == '__main__':
    x = ['终于快弄好了', '今天天气不错', '你在干什么']
    from transformers import BertTokenizer
    model = bert_douban(model_config.bert_type, 100, 3, 1, 0.5)

    tokenizer = BertTokenizer.from_pretrained(model_config.bert_type)
    t = tokenizer(x, max_length=10, truncation=True, padding='max_length')
    # logits = model((t['inpu'])

