from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch
from torch.utils.data import Dataset, random_split, DataLoader

import pandas as pd
import os

from huggingface_hub import login

from utls import *

class SmolLm2135M:

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def exportKey(self):
        path = HF_KEY_PATH
        path = os.path.join(path, 'key')
        with open(path, 'rt') as f:
            key = f.read().strip()
        os.environ['HF_TOKEN'] = key
        login()
        return key

    def setModel(self):
        device = self.device
        model_name = self.model_path
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            print("Chat template already exists.")
        else:
            print("Chat template does not exist.")
            model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

        return model, tokenizer

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


if __name__ == "__main__":

        data_path = DATA_PATH
        data_path = os.path.join(data_path, 'qc.xlsx')
        dataset = QuantumQA(data_path)
        print(dataset[0])