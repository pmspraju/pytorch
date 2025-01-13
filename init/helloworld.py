import os.path
import torch
import numpy as np
print(f"Hello, PyTorch {torch.__version__}!")
print(f"Hello, NumPy {np.__version__}!")
print(f"Your current working directory is {os.getcwd()}")
print(f"Cuda is available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Cuda version: {torch.version.cuda}")
    print(f"Cuda device count: {torch.cuda.device_count()}")
    print(f"Cuda device name: {torch.cuda.get_device_name()}")