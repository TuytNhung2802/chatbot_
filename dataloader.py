import torch

import pytorch_lightning as pl

from transformers import T5Tokenizer
from torch.utils.data import Dataset,DataLoader


class ChatBotDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.data)


    def __getitem__(self,index):
        data_row = self.data.iloc[index]

        source_encoding = self.tokenizer.encode_plus(
            str(data_row["state"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoding = self.tokenizer.encode_plus(
            str(data_row["content"]),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100
        return dict(
            question=str(data_row["state"]),
            context=str(data_row["content"]),
            input_ids=source_encoding["input_ids"].flatten(),
            attention_mask=source_encoding["attention_mask"].flatten(),
            decoder_attention_mask = target_encoding["attention_mask"].flatten(),
            labels=labels.flatten()
        )


class ChatBotDataModel(pl.LightningDataModule):
    def __init__(self, device, model_name, train, test, batch_size=16, max_length=256):
        super().__init__()
        self.train = train
        self.test = test
        self.tokenizer  = T5Tokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device


    def setup(self,stage=None):
        self.train_dataset = ChatBotDataset(self.train, self.tokenizer, self.max_length)
        self.test_dataset = ChatBotDataset(self.test, self.tokenizer, self.max_length)


    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.test_dataset,batch_size=self.batch_size)