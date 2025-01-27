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

    # test the base model
    finetuner = FinetuneSmolLLM2135M()
    #finetuner.testbasemodel()


    # fine tune:Train the model
    #finetuner.finetune(train_ds, test_ds)

    # Test the fine tuned model
    saved_finetuner = FinetuneSmolLLM2135M(FINE_TUNE_MODEL_PATH)
    saved_finetuner.testfinetunedmodel()














