import torch
from torch import dtype
from torch.utils.data import DataLoader, Dataset

# create datasets from tensors
t = torch.arange(1, 10, dtype=torch.float32)
data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader):
    print(f"Batch {i}:", batch)

# Joint datasets by subclassing
torch.manual_seed(1)
t_features = torch.rand([4, 3], dtype=torch.float32)
t_labels = torch.arange(4)

# A subclassed custom dataset should contain the following methods:
# __init__(): reading existing arrays, filtering
# __len__(): Return the length of the dataset
# __getitem__(): Return a sample from the dataset at a specific index

class joindataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

f_l = joindataset(t_features, t_labels)
for i, example in enumerate(f_l):
    print(f"Example {i}: Feature: {example[0]} label: {example[1]}")