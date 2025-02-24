from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from datasets import load_dataset

from pprint import pprint

import pandas as pd
import os

from huggingface_hub import login

from utls import *

class TokenizeDataset(Dataset):

    def __init__(self, dataset, tokenizer, verify=False):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.verify = verify

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if idx==0:
            print(item)
        if self.verify:
            input_text = self.tokenizer.apply_chat_template(item, tokenize=False)
            print("Conversation with template:", input_text)

        input_text = self.tokenizer.apply_chat_template(
            item, tokenize=True, add_generation_prompt=True
        )
        return input_text

class PrepareSmolTalk():

    def __init__(self, dataset):
        self.dataset = dataset

    def data_genrator(self):
        for item in self.dataset:

            # extract the prompt
            prompt = ' '
            for x in item['chosen']:
                if x['role'] == 'user':
                    prompt = x['content']
                    prompt = prompt_template(prompt)
                    break
            # yield the instance as per chat template
            yield [{'prompt': prompt,
                    'chosen':item['chosen'],
                    'rejected':item['rejected']
                    }]

    def get_pairs(self):
        generator = self.data_genrator()
        while True:
            try:
                message = next(generator)
                yield message
            except StopIteration:
                break

class SmolTalkDataset(Dataset):

    def __init__(self, hf_ds_path, hf_ds_name="default", type='train'):
        ds = load_dataset(path=hf_ds_path, name=hf_ds_name)
        self.dataset = ds[type]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]#['chosen']
        return sample

if __name__ == "__main__":

        #data_path = DATA_PATH
        #data_path = os.path.join(data_path, 'qc.xlsx')
        #dataset = QuantumQA(data_path)
        #print(dataset[0])

        hf_ds_path = "trl-lib/ultrafeedback_binarized"
        #hf_ds_name = "everyday-conversations"

        dataset = SmolTalkDataset(hf_ds_path)
        genn = PrepareSmolTalk(dataset).get_pairs()
        dataset = GeneratorDataset(genn)
        #pprint(dataset[10])

        #data_loader = DataLoader(dataset, batch_size=2, drop_last=False)
        #for i, batch in enumerate(data_loader):
        #    print(f"Batch {i}:", batch)
        #    break
