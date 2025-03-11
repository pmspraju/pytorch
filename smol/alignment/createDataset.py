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

    def __init__(self, gen, tokenizer, verify=False):
        self.data = list(gen)
        self.tokenizer = tokenizer
        self.verify = verify

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        items = self.data[idx]
        item = items[0]
        print(item['prompt'])

        prompt   = item['prompt']
        chosen   = item['chosen']
        rejected = item['rejected']

        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )

        chosen_tokens = self.tokenizer.apply_chat_template(
            chosen, tokenize=True, add_generation_prompt=True
        )

        rejected_tokens = self.tokenizer.apply_chat_template(
            rejected, tokenize=True, add_generation_prompt=True
        )

        tokenized = [{
            'prompt': prompt_tokens,
            'chosen': chosen_tokens,
            'rejected': rejected_tokens
        }]

        return tokenized

class PrepareSmolTalk():

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def create_tokens(self, data):
        tokens = self.tokenizer.apply_chat_template(
            data, tokenize=True, add_generation_prompt=True
        )
        return tokens

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
            yield {'prompt': prompt,
                    'chosen':item['chosen'],
                    'rejected':item['rejected']
                    }

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

class HFGeneratorDataset:
    def __init__(self, huggingface_dataset, tokenizer):
        self.dataset = huggingface_dataset
        self.tokenizer = tokenizer

    def prepare(self, example):
        prompt = ' '
        for x in example['chosen']:
            if x['role'] == 'user':
                prompt = x['content']
                prompt = prompt_template(prompt)
                break

        prompt_tokens = self.tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        chosen_tokens = self.tokenizer.apply_chat_template(
            example['chosen'], tokenize=True, add_generation_prompt=True
        )
        reject_tokens = self.tokenizer.apply_chat_template(
            example['rejected'], tokenize=True, add_generation_prompt=True
        )

        # yield the instance as per chat template
        return {'prompt': prompt_tokens,
                'chosen': chosen_tokens,
                'rejected': reject_tokens
                }

    def __iter__(self):
        # Reinitialize the generator for each worker process
        for item in self.dataset:
            yield self.prepare(item)

def apply_token(item, tokenizer):
    prompt = ' '
    for x in item['chosen']:
        if x['role'] == 'user':
            prompt = x['content']
            #prompt = prompt_template(prompt)
            break

    chosen = ' '
    for x in item['chosen']:
        if x['role'] == 'assistant':
            chosen = x['content']
            break

    rejected = ' '
    for x in item['rejected']:
        if x['role'] == 'assistant':
            rejected = x['content']
            break

    # prompt_tokens = tokenizer.apply_chat_template(
    #         prompt, tokenize=True, add_generation_prompt=True
    #     )
    # chosen_tokens = tokenizer.apply_chat_template(
    #         item['chosen'], tokenize=True, add_generation_prompt=True
    #     )
    # reject_tokens = tokenizer.apply_chat_template(
    #     item['rejected'], tokenize=True, add_generation_prompt=True
    # )
    prompt_input_ids = prompt
    chosen_input_ids = chosen
    rejected_input_ids = rejected

    # yield the instance as per chat template
    result = {'prompt': prompt_input_ids,
              'chosen': chosen_input_ids,
              'rejected': rejected_input_ids
             }

    return result

if __name__ == "__main__":

        #data_path = DATA_PATH
        #data_path = os.path.join(data_path, 'qc.xlsx')
        #dataset = QuantumQA(data_path)
        #print(dataset[0])

        hf_ds_path = "trl-lib/ultrafeedback_binarized"
        #hf_ds_name = "everyday-conversations"

        hf_ds_name = "default"
        ds = load_dataset(path=hf_ds_path, name=hf_ds_name)
        ds = ds['train']
        dsmap = ds.map(apply_token, fn_kwargs={'tokenizer':'tokenizer'})
        print(dsmap[0])

        #dataset = SmolTalkDataset(hf_ds_path)
        #genn = PrepareSmolTalk(dataset).get_pairs()
        #dataset = GeneratorDataset(genn)
        #pprint(dataset[0])

        #from torch.utils.data import Subset
        #first_10_dataset = Subset(dataset, indices=list(range(10)))
        #pprint(first_10_dataset[11])
        #data_loader = DataLoader(dataset, batch_size=2, drop_last=False)
        #for i, batch in enumerate(data_loader):
        #    print(f"Batch {i}:", batch)
        #    break
