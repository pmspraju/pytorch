from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from datasets import load_dataset

import pandas as pd
import os

from huggingface_hub import login

from utls import *

class QuantumQA(Dataset):

    def __init__(self, data_path):
        df = pd.read_excel(data_path)
        self.question = df['Title'].tolist()
        self.answer = df['Answer'].tolist()

    def __len__(self):
        return len(self.question)

    def __getitem__(self, idx):
        message = buildSmolLLM135Message(self.question[idx], self.answer[idx])
        return message

class TokenizeDataset(Dataset):

    def __init__(self, dataset, tokenizer, verify=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.verify = verify

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        if self.verify:
            input_text = self.tokenizer.apply_chat_template(item, tokenize=False)
            print("Conversation with template:", input_text)

        input_text = self.tokenizer.apply_chat_template(
            item, tokenize=True, add_generation_prompt=True
        )
        return input_text

class PrepareDataset:

    def __init__(self, ds, train_size=0.8, test_size=0.2, batch_size=1, shuffle=False, seed=42):
        self.dataset = ds
        self.train_size = train_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def split(self):
        train_size = int(len(self.dataset) * self.train_size)
        test_size = len(self.dataset) - train_size
        gen = torch.Generator().manual_seed(self.seed)
        return random_split(self.dataset, [train_size, test_size], generator=gen)

    def dataloader(self):
        train_dataset, test_dataset = self.split()
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=self.shuffle)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # return train_loader, test_loader
        return train_dataset, test_dataset

class SmolTalkDataset(Dataset):

    def __init__(self, hf_ds_path, hf_ds_name, type='train'):
        ds = load_dataset(path=hf_ds_path, name=hf_ds_name)
        self.dataset = ds[type]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]['messages']

class PrepareSmolTalk():

    def __init__(self, dataset):
        self.dataset = dataset

    def data_genrator(self):
        for item in self.dataset:
            for i in item:
                yield i['content']

    def get_pairs(self):
        generator = self.data_genrator()
        while True:
            try:
                message = buildSmolLLM135Message(next(generator), next(generator))
                yield message
            except StopIteration:
                break

if __name__ == "__main__":

        #data_path = DATA_PATH
        #data_path = os.path.join(data_path, 'qc.xlsx')
        #dataset = QuantumQA(data_path)
        #print(dataset[0])

        hf_ds_path = "HuggingFaceTB/smoltalk"
        hf_ds_name = "everyday-conversations"

        dataset = SmolTalkDataset(hf_ds_path, hf_ds_name)
        genn = PrepareSmolTalk(dataset).get_pairs()
        dataset = GeneratorDataset(genn)
        print(dataset[0])
