import os.path
import torch

from createDataset import *
from createModel import *
from finetune import FinetuneSmolLLM2135M
from utls import *

def sft_quantum():
    # set the dataset
    datasetQA = QuantumQA(os.path.join(DATA_PATH, 'qc.xlsx'))
    # print("Original conversation:", datasetQA[0])
    datasetTOK = TokenizeDataset(dataset=datasetQA, tokenizer=tokenizer)
    # print("Conversation tokenized:", datasetTOK[0])
    # print("Conversation decoded:", tokenizer.decode(token_ids=datasetTOK[0]))

    # Split the dataset
    train_ds, test_ds = PrepareDataset(datasetTOK, ).dataloader()

    return train_ds, test_ds

def sft_smalltalk():
    hf_ds_path = "HuggingFaceTB/smoltalk"
    hf_ds_name = "everyday-conversations"

    type = 'train'
    dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, type)
    genn = PrepareSmolTalk(dataset).get_pairs()
    train_ds = GeneratorDataset(genn)
    train_ds = TokenizeDataset(dataset=train_ds, tokenizer=tokenizer)

    type = 'test'
    dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, type)
    genn = PrepareSmolTalk(dataset).get_pairs()
    test_ds = GeneratorDataset(genn)
    test_ds = TokenizeDataset(dataset=test_ds, tokenizer=tokenizer)

    return train_ds, test_ds

if __name__ == "__main__":

    # Set the model
    model, tokenizer = SmolLm2135M(MODEL_SMOLLM2_135M_PATH).setModel()

    # Set the quantum dataset
    #train_ds, test_ds = sft_quantum()
    #print(train_ds[0])

    # Set the small talk dataset
    train_ds, test_ds = sft_smalltalk()
    print(train_ds[0])

    print("Length of train dataset:", len(train_ds))
    print("Length of test dataset:", len(test_ds))
    print("Number of steps per epoch:", len(train_ds) // BATCH_SIZE)
    #print("Conversation decoded:", tokenizer.decode(token_ids=train_ds[0]))

    # fine tune:Train the model
    # finetuner.finetune(train_ds, test_ds)

    # Print memory snapshot
    #pickle_path = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135Mfinetune'
    #pickle_path = os.path.join(pickle_path, 'memory_snapshot.pkl')
    #printMemorySnapshot(pickle_path)

    # test the base model
    finetuner = FinetuneSmolLLM2135M()
    prompt = "Write a haiku about programming"
    print('**** Base model response ****')
    finetuner.testbasemodel(prompt)
    print('*****************************')

    # Test the fine tuned model
    saved_finetuner = FinetuneSmolLLM2135M(FINE_TUNE_MODEL_PATH)
    prompt = "Write a haiku about programming"
    print('**** Fine-tuned model response ****')
    saved_finetuner.testfinetunedmodel(prompt)
    print('*****************************')














