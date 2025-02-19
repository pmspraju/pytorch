import os.path
import torch

from createDataset import *
from createModel import *
from createModel import FinetuneSmolLLM2135M
from utls import *
from pprint import pprint

def dpo_smalltalk():
    hf_ds_path = "trl-lib/ultrafeedback_binarized"
    hf_ds_name = None

    type = 'train'
    dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, type)
    #genn = PrepareSmolTalk(dataset).get_pairs()
    #train_ds = GeneratorDataset(genn)
    train_ds = TokenizeDataset(dataset=dataset, tokenizer=tokenizer)

    type = 'test'
    dataset = SmolTalkDataset(hf_ds_path, hf_ds_name, type)
    #genn = PrepareSmolTalk(dataset).get_pairs()
    #test_ds = GeneratorDataset(genn)
    test_ds = TokenizeDataset(dataset=dataset, tokenizer=tokenizer)

    return train_ds, test_ds

if __name__ == "__main__":

    # Set the model
    model, tokenizer = SmolLm2135M(MODEL_SMOLLM2_135M_I_PATH).setModel()

    # empty the cache
    torch.cuda.empty_cache()

    # Set the small talk dataset
    train_ds, test_ds = dpo_smalltalk()

    print("Length of train dataset:", len(train_ds))
    print("Length of test dataset:", len(test_ds))
    print("Number of steps per epoch:", len(train_ds) // BATCH_SIZE)
    #print("Conversation decoded:", tokenizer.decode(token_ids=train_ds[0]))

    kwargs = {
        'finetune_name': "SmolLM2-FT-DPO",
        'finetune_tags': ['smol-course', 'DPO']
    }
    # fine tune:Train the model
    finetuner = FinetuneSmolLLM2135M(MODEL_SMOLLM2_135M_I_PATH, **kwargs)
    finetuner.finetune(train_ds, test_ds)

    # Print memory snapshot
    pickle_path = FINE_TUNE_MODEL_PATH
    pickle_path = os.path.join(pickle_path, 'memory_snapshot.pkl')
    printMemorySnapshot(pickle_path)

    # test the base model
    # finetuner = FinetuneSmolLLM2135M()
    # prompt = "Write a haiku about programming"
    # print('**** Base model response ****')
    # finetuner.testbasemodel(prompt)
    # print('*****************************')

    # Test the fine tuned model
    # saved_finetuner = FinetuneSmolLLM2135M(FINE_TUNE_MODEL_PATH)
    # prompt = "Write a haiku about programming"
    # print('**** Fine-tuned model response ****')
    # saved_finetuner.testfinetunedmodel(prompt)
    # print('*****************************')














