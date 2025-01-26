import os

# Data path
DATA_PATH = r'/home/nachiketa/Documents/Workspaces/pytorch/smol/data'

# HuggingFaceTB/SmolLM2-135M
MODEL_SMOLLM2_135M_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135M'

# Hugging face key path
HF_KEY_PATH = r'/home/nachiketa/Documents/Keys/hugging_face'

# fine tune model path
FINE_TUNE_MODEL_PATH = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135Mfinetune'
CHECK_POINT = "SmolLM2-FT-QuantumQA"
FINE_TUNE_MODEL_PATH = os.path.join(FINE_TUNE_MODEL_PATH, CHECK_POINT)


TRAIN_SIZE = 0.8
TEST_SIZE  = 0.2
BATCH_SIZE = 2
# One step = processing one batch of data
# One epoch = processing the entire dataset once
# 1000 training examples
# Batch size of 10
# One epoch would take 100 steps (1000/10)

from pynvml import *

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