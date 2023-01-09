import torch
from torch import nn
from transformers import BertModel
import torch.nn.functional as F


class Bert(nn.Module):
    def __init__(self, dropout=0.5):
        super(Bert, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, mask):
        # token级，句子级embedding
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   attention_mask=mask,
                                                   return_dict=False)
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        return sequence_output, pooled_output


class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()
        self.fc = nn.Linear(768, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, bert_out):
        fc_out = self.fc(bert_out)
        out = self.softmax(fc_out)
        return out


class TextCNN(nn.Module):
    def __init__(self, device, dropout=0.5):
        super(TextCNN, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.convs = [nn.Conv2d(1, 1, (4, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (4, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (4, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (3, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (3, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (3, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (2, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (2, 768), stride=(1,)).to(device),
                      nn.Conv2d(1, 1, (2, 768), stride=(1,)).to(device)]
        self.FC = nn.Linear(9, 2)
        self.softmax = nn.Softmax(dim=1)

    def conv_pool(self, x, conv):
        # (N_batch, C, W, H) -> (N_batch, C, out, 1)
        x = conv(x)
        x = F.relu(x, inplace=True)
        # (N_batch, C, out, 1) -> (N_batch, C, out)
        x = x.squeeze(3)
        kernel_size = x.shape[2]  # out
        x = F.max_pool1d(x, kernel_size)  # (N_batch, C, 1)
        return x

    def forward(self, bert_out):
        # (N_batch, W, 768) -> (N_batch, 1, W, 768)
        bert_out = bert_out.unsqueeze(1)
        # (N_batch, C, FC_in)
        cnn_out = torch.cat([self.conv_pool(bert_out, conv) for conv in self.convs], 2)
        # C = 1
        # (N_batch, C, FC_in) -> (N_batch, FC_in)
        cnn_out = cnn_out.squeeze(1)
        fc_in = self.dropout(cnn_out)
        fc_out = self.FC(fc_in)
        out = self.softmax(fc_out)
        return out
