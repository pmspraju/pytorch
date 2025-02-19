import math

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer, setup_chat_format

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import setup_chat_format
from huggingface_hub import login

from trl import DPOTrainer, DPOConfig

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
            pretrained_model_name_or_path=model_name,
            torch_dtype=torch.float32,
        ).to(device)
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)
        tokenizer.pad_token = tokenizer.eos_token

        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            print("Chat template already exists.")
        else:
            print("Chat template does not exist.")
            model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

        return model, tokenizer

class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print ('End of epoch memory usage')
        print_gpu_utilization()

        # record memory snapshot
        #torch.cuda.memory._dump_snapshot(f"memory_epoch_{state.epoch}.pkl")

        return
        # Clear the cache after every N epochs
        N = 3
        if (state.epoch + 1) % N == 0:
            print('Before clearing cache:')
            print_gpu_utilization()

            print(f"Clearing GPU cache after epoch {state.epoch}...")
            torch.cuda.empty_cache()

            print('After clearing cache:')
            print_gpu_utilization()

class FinetuneSmolLLM2135M:
    def __init__(self, model_path=MODEL_SMOLLM2_135M_I_PATH, **kwargs):
        # clear the cache before fine tuning
        torch.cuda.empty_cache()

        # set the device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # set the model
        self.baseModel, self.tokenizer = SmolLm2135M(model_path).setModel()

        for key, value in kwargs.items():
            if key == 'finetune_name':
                self.finetune_name = value
            if key == 'finetune_tags':
                self.finetune_tags = value

    def setDPOConfig(self):
        if self.finetune_name:
            finetune_name = self.finetune_name
        else:
            finetune_name = "SmolLM2-FT-DPO"

        # Configure the SFTTrainer
        self.dpo_config = DPOConfig(
            # Training batch size per GPU
            per_device_train_batch_size=4,
            # Number of updates steps to accumulate before performing a backward/update pass
            # Effective batch size = per_device_train_batch_size * gradient_accumulation_steps
            gradient_accumulation_steps=4,
            # Saves memory by not storing activations during forward pass
            # Instead recomputes them during backward pass
            gradient_checkpointing=True,
            # Base learning rate for training
            learning_rate=5e-5,
            # Learning rate schedule - 'cosine' gradually decreases LR following cosine curve
            lr_scheduler_type="cosine",
            # Total number of training steps
            max_steps=2,
            # Disables model checkpointing during training
            save_strategy="no",
            # How often to log training metrics
            logging_steps=1,
            # Directory to save model outputs
            output_dir="smol_dpo_output",
            # Number of steps for learning rate warmup
            warmup_steps=100,
            # Use bfloat16 precision for faster training
            bf16=True,
            # Disable wandb/tensorboard logging
            report_to="none",
            # Keep all columns in dataset even if not used
            remove_unused_columns=False,
            # Enable MPS (Metal Performance Shaders) for Mac devices
            #use_mps_device=self.device,
            # Model ID for HuggingFace Hub uploads
            hub_model_id=finetune_name,
            # DPO-specific temperature parameter that controls the strength of the preference model
            # Lower values (like 0.1) make the model more conservative in following preferences
            beta=0.1,
            # Maximum length of the input prompt in tokens
            max_prompt_length=1024,
            # Maximum combined length of prompt + response in tokens
            max_length=1536,
        )

    def testbasemodel(self, prompt):
        #prompt = "Write a haiku about programming"
        #prompt = "Can a Turing machine simulate a quantum computer"

        # Format with template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

        # Generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.baseModel.generate(**inputs, max_new_tokens=100)
        print("Before training:")
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def finetune(self, train_ds, test_ds):

        # Set the SFTConfig
        self.setDPOConfig()

        # Initialize the DPOTrainer
        trainer = DPOTrainer(
            # The model to be trained
            model=self.baseModel,
            # Training configuration from above
            args=self.dpo_config,
            # Dataset containing preferred/rejected response pairs
            train_dataset=train_ds,
            eval_dataset=test_ds,
            # Tokenizer for processing inputs
            processing_class=self.tokenizer,
            callbacks=[ClearCacheCallback()],
            # DPO-specific temperature parameter that controls the strength of the preference model
            # Lower values (like 0.1) make the model more conservative in following preferences
            # beta=0.1,
            # Maximum length of the input prompt in tokens
            # max_prompt_length=1024,
            # Maximum combined length of prompt + response in tokens
            # max_length=1536,
        )

        # print the gpu utilization
        print_gpu_utilization()

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(f"./{self.finetune_name}")

    def testfinetunedmodel(self, prompt):
        #prompt = "Write a haiku about programming"
        #prompt = "Can a Turing machine simulate a quantum computer"

        # Format with template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # Generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.baseModel.generate(**inputs, max_new_tokens=100)
        print("After training:")
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))