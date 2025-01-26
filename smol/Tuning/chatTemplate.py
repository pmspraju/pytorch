import os.path
import torch


from createDataset import SmolLm2135M, QuantumQA, TokenizeDataset, PrepareDataset
from finetune import FinetuneSmolLLM2135M
from utls import *

if __name__ == "__main__":

    # Set the model
    model, tokenizer = SmolLm2135M(MODEL_SMOLLM2_135M_PATH).setModel()

    # set the dataset
    datasetQA = QuantumQA(os.path.join(DATA_PATH, 'qc.xlsx'))
    #print("Original conversation:", datasetQA[0])
    datasetTOK = TokenizeDataset(dataset=datasetQA, tokenizer=tokenizer)
    #print("Conversation tokenized:", datasetTOK[0])
    #print("Conversation decoded:", tokenizer.decode(token_ids=datasetTOK[0]))

    # Split the dataset
    train_ds, test_ds = PrepareDataset(datasetTOK,).dataloader()
    print("Length of train dataset:", len(train_ds))
    print("Length of test dataset:", len(test_ds))
    print("Number of steps per epoch:", len(train_ds) // BATCH_SIZE)
    #print("Conversation decoded:", tokenizer.decode(token_ids=train_ds[0]))

    # set the finetuner
    #finetuner = FinetuneSmolLLM2135M()
    #finetuner.testbasemodel()

    # emtpy the cache
    torch.cuda.empty_cache()

    # check the gpu memory
    if torch.cuda.is_available():
        gpu_id = 0  # Replace with the GPU ID you want to check
        free_mem, total_mem = torch.cuda.mem_get_info(device=f'cuda:{gpu_id}')
        print(f"Free memory: {free_mem / (1024 ** 2):.2f} MB")
        print(f"Total memory: {total_mem / (1024 ** 2):.2f} MB")
    else:
        print("CUDA is not available.")

    # fine tune:Train the model
    #finetuner.finetune(train_ds, test_ds)

    # Test the model
    saved_finetuner = FinetuneSmolLLM2135M(FINE_TUNE_MODEL_PATH)
    saved_finetuner.testfinetunedmodel()














