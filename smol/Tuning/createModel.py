from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from huggingface_hub import login

from utls import *

class SmolLm2135M:

    def __init__(self, model_path):
        self.model_path = model_path
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

    def exportKey(self):
        path = HF_KEY_PATH
        path = os.path.join(path, 'key')
        with open(path, 'rt') as f:
            key = f.read().strip()
        os.environ['HF_TOKEN'] = key
        login()
        return key

    def setModel(self):
        device = self.device
        model_name = self.model_path
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            print("Chat template already exists.")
        else:
            print("Chat template does not exist.")
            model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

        return model, tokenizer
