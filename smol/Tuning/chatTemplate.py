import os.path

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
import torch

from huggingface_hub import login

def exportKey():
    path = r'/home/nachiketa/Documents/Keys/hugging_face'
    path = os.path.join(path, 'key')
    with open(path, 'rt') as f:
        key = f.read().strip()
    os.environ['HF_TOKEN'] = key
    return key

def setDevice():

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return device

def setModel(model_name):
    device = setDevice()
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
    model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    return model, tokenizer

if __name__ == "__main__":

    # Get the key and login
    key = exportKey()
    print(key)
    login()
    print("Logged in successfully.")

    # Set the model
    #model_name = "HuggingFaceTB/SmolLM2-135M"
    model_path = r'/home/nachiketa/Documents/Workspaces/checkpoints/smolLM2-135M'
    model_name = model_path
    model, tokenizer = setModel(model_name)

    # Define message format for the base model SmolLM2-135M
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {
            "role": "assistant",
            "content": "I'm doing well, thank you! How can I assist you today?",
        },
    ]

    # apply chat template to messages without tokenization
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    print("Conversation with template:", input_text)

    # Check the decode with tokenization
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )

    print("Conversation decoded:", tokenizer.decode(token_ids=input_text))











