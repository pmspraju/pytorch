import os
import pickle
import numpy as np

# Data path
DATA_PATH = r'/home/nachiketa/Documents/Workspaces/pytorch/smol/data'

# HuggingFaceTB/SmolLM2-135M
MODEL_SMOLLM2_135M_I_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135M-Instruct'

# Hugging face key path
HF_KEY_PATH = r'/home/nachiketa/Documents/Keys/hugging_face'

# fine tune model path
FINE_TUNE_MODEL_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135Mfinetune_DPO'
#CHECK_POINT = "checkpoint-18396"
#FINE_TUNE_MODEL_PATH = os.path.join(FINE_TUNE_MODEL_PATH, CHECK_POINT)
#FINE_TUNE_MODEL_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/temp'


TRAIN_SIZE = 0.8
TEST_SIZE  = 0.2
BATCH_SIZE = 2
# One step = processing one batch of data
# One epoch = processing the entire dataset once
# 1000 training examples
# Batch size of 10
# One epoch would take 100 steps (1000/10)

from pynvml import *
import torch
from torch.utils.data import Dataset

class GeneratorDataset(Dataset):
    def __init__(self, generator):
        # Convert generator to list for random access
        self.data = list(generator)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def buildSmolLLM135Message(question, answer):
    message = [
                {
                    "role": "user",
                    "content": question
                },
                {
                    "role": "assistant",
                    "content": answer
                }
            ]
    return message

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def avbl_memory():
    # check the gpu memory
    if torch.cuda.is_available():
        gpu_id = 0  # Replace with the GPU ID you want to check
        free_mem, total_mem = torch.cuda.mem_get_info(device=f'cuda:{gpu_id}')
        print(f"Free memory: {free_mem / (1024 ** 2):.2f} MB")
        print(f"Total memory: {total_mem / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available.")

def printMemorySnapshot(pickle_path):
    with open(pickle_path, "rb") as f:
       snapshot = pickle.load(f)

    print(snapshot)  # Prints memory usage statistics

def prompt_template(prompt):
    # Base prompt template
    prompt_template = [
            {
                "role": "assistant",
                "content": "You are bot, a helpful AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

    return prompt_template

def checkMap(ds):
    if hasattr(ds, 'map'):
        print("yes map exists")
    else:
        print("train_dataset does not support .map(). Ensure it's a datasets.Dataset object.")

data = [
    {
        "prompt": [
            {"content": "You are bot, a helpful AI assistant.", "role": "assistant"},
            {"content": "you are ai", "role": "user"}
        ],
        "chosen": [
            {"content": "you are AI", "role": "user"},
            {"content": "you are AI", "role": "assistant"}
        ],
        "rejected": [
            {"content": "you are AI", "role": "user"},
            {"content": "you are AI", "role": "assistant"}
        ]
    },
    {
        "prompt": [
            {"content": "What is the capital of France?", "role": "user"},
            {"content": "The capital of France is Paris.", "role": "assistant"}
        ],
        "chosen": [
            {"content": "The capital of France is Paris.", "role": "assistant"},
            {"content": "Thank you!", "role": "user"}
        ],
        "rejected": [
            {"content": "The capital of France is Lyon.", "role": "assistant"},
            {"content": "Are you sure?", "role": "user"}
        ]
    }
]

class CreatetempDataset(Dataset):
    def __init__(self, data, tokeninzer):
        self.data = data
        self.tokenizer = tokeninzer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        #prompt = item['prompt']
        #chosen = item['chosen']
        #rejected = item['rejected']
        prompt = " ".join([turn["content"] for turn in item["prompt"]])
        chosen = " ".join([turn["content"] for turn in item["chosen"]])
        rejected = " ".join([turn["content"] for turn in item["rejected"]])
        print({'prompt':prompt, 'chosen':chosen, 'rejected':rejected})

        # Tokenize the data
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        chosen_tokens = self.tokenizer(chosen, return_tensors="pt", padding=True, truncation=True)
        rejected_tokens = self.tokenizer(rejected, return_tensors="pt", padding=True, truncation=True)

        # Transform the data as needed
        # For example, you can convert text to numerical data (tokenization) here
        return {
            'prompt': prompt_tokens,
            'chosen': chosen_tokens,
            'rejected': rejected_tokens
        }

def testTokenize(tokeninzer):
    tds = CreatetempDataset(data, tokeninzer)
    return  tds
