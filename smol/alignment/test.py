from datasets import IterableDataset
import torch
from utls import *

# PyTorch dataset generator
class MyCustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Example PyTorch dataset
pytorch_data = [
    {"prompt": "What is the capital of France?", "chosen": "Paris", "rejected": "Lyon"},
    {"prompt": "What is 2 + 2?", "chosen": "4", "rejected": "5"}
]

#pytorch_dataset = MyCustomDataset(pytorch_data)
pytorch_dataset = MyCustomDataset(data)

# Create a generator function
def generator():
    for i in range(len(pytorch_dataset)):
        yield pytorch_dataset[i]

# Convert generator to Hugging Face IterableDataset
print(type(generator()))
hf_dataset = IterableDataset.from_generator(generator)
checkMap(hf_dataset)

# Verify the dataset
for example in hf_dataset:
    print(example)

###############################################################
#split_type = 'train'
    #dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, split_type)
    #genn = PrepareSmolTalk(dataset, tokenizer).data_genrator()  # .get_pairs()
    #genn = data_gen(dataset)
    #print(next(genn))
    # genn = GeneratorDataset(genn)
    # train_ds = TokenizeDataset(genn, tokenizer=tokenizer)
    #train_ds = IterableDataset.from_generator(genn)

    # split_type = 'test'
    # dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, split_type)
    # genn = PrepareSmolTalk(dataset, tokenizer).get_pairs()
    # # genn = GeneratorDataset(genn)
    # # test_ds = TokenizeDataset(genn, tokenizer=tokenizer)
    # test_ds = IterableDataset.from_generator(genn)

    # Set the small talk dataset
    #checkMap(train_ds)
    #print(train_ds[0])
    #pprint(detokenize(tokenizer, test_ds[0]))