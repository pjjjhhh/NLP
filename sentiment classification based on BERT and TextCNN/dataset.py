import torch
from torch.utils.data import Dataset
import pandas as pd
from transformers import BertTokenizer


class TCDataset(Dataset):
    def __init__(self, data_path, is_train=True, test=False):
        super(TCDataset, self).__init__()
        self.is_train = is_train
        self.test = test
        if not self.test:
            if self.is_train:
                self.data_path = data_path + '/train.csv'
            else:
                self.data_path = data_path + '/val.csv'
        else:
            self.data_path = data_path + '/test.csv'
        self.data_df = pd.read_csv(self.data_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=False)
        self.max_length = 19 + 2

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        label = list(self.data_df.iloc[:, 0])[idx]
        text = list(self.data_df.iloc[:, 1])[idx]

        text_input = self.tokenizer(text,
                                    padding='max_length',
                                    max_length=self.max_length,
                                    truncation=True,
                                    return_tensors="pt")
        input_ids = torch.squeeze(text_input['input_ids'], 0)
        token_type_ids = torch.squeeze(text_input['token_type_ids'], 0)
        attention_mask = torch.squeeze(text_input['attention_mask'], 0)

        target = torch.zeros([2])
        target[label] = 1
        return_tuple = (input_ids, token_type_ids, attention_mask, target)
        return return_tuple
