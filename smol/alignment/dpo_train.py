import os.path
import torch
from torch.utils.data import Subset
from datasets import Dataset, IterableDataset

from createDataset import *
from createModel import *
from createModel import FinetuneSmolLLM2135M

from utls import *
from pprint import pprint

def data_gen(ds):
    for item in ds:

        # extract the prompt
        prompt = ' '
        for x in item['chosen']:
            if x['role'] == 'user':
                prompt = x['content']
                prompt = prompt_template(prompt)
                break
        # yield the instance as per chat template
        yield {'prompt': prompt,
               'chosen': item['chosen'],
               'rejected': item['rejected']
               }

def detokenize(tokenizer, tokens):
    decoded = {}
    for key in tokens:
        decoded[key] = tokenizer.decode(tokens[key], skip_special_tokens=True)

    return decoded

if __name__ == "__main__":

    # Set the model
    model, tokenizer = SmolLm2135M(MODEL_SMOLLM2_135M_I_PATH).setModel()

    # empty the cache
    torch.cuda.empty_cache()

    hf_ds_path = "trl-lib/ultrafeedback_binarized"
    hf_ds_name = "default"

    ds = load_dataset(path=hf_ds_path, name=hf_ds_name)

    ### Lazy loading - did not work because of num_proc issue in DPOTrainer map()
    # ds_train = ds['train']
    # dsgen_train = HFGeneratorDataset(ds_train, tokenizer).__iter__
    # train_ds = IterableDataset.from_generator(dsgen_train)

    # ds_test = ds['test']
    # dsgen_test = HFGeneratorDataset(ds_test, tokenizer).__iter__
    # test_ds = IterableDataset.from_generator(dsgen_test)

    # checkMap(test_ds)
    # for example in test_ds:
    #     print(example)
    #     break

    ### Eager loading
    ds_train = ds['train']
    ds_train_token = ds_train.map(apply_token,
                            fn_kwargs={'tokenizer':tokenizer},
                            keep_in_memory=True, # Ensures the mapping occurs in memory
                            load_from_cache_file= False#False # Avoids cache issues
                            )
    train_ds = ds_train_token.remove_columns(["score_chosen", "score_rejected"])
    # for example in ds_train_token:
    #     print(example.keys())
    #     break

    ds_test = ds['test']
    ds_test_token = ds_test.map(apply_token,
                                fn_kwargs={'tokenizer': tokenizer},
                                keep_in_memory=True,
                                load_from_cache_file=False
                                )
    test_ds = ds_test_token.remove_columns(["score_chosen", "score_rejected"])
    # for example in test_ds:
    #     pprint(example)
    #     break

    ## Testing by creating subsets
    #train_ds = Subset(train_ds, indices=list(range(1)))
    #test_ds = Subset(test_ds, indices=list(range(1)))
    #print(train_ds[0])

    #print("Length of train dataset:", len(train_ds))
    #print("Length of test dataset:", len(test_ds))
    #print("Number of steps per epoch:", len(train_ds) // BATCH_SIZE)

    kwargs = {
        'finetune_name': "SmolLM2-FT-DPO",
        'finetune_tags': ['smol-course', 'DPO']
    }

    # fine tune:Train the model
    finetuner = FinetuneSmolLLM2135M(MODEL_SMOLLM2_135M_I_PATH, **kwargs)
    finetuner.finetune(train_ds, test_ds)

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