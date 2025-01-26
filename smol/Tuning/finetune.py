import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format

from utls import *
from createDataset import SmolLm2135M

class ClearCacheCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"Clearing GPU cache after epoch {state.epoch}...")
        torch.cuda.empty_cache()

class FinetuneSmolLLM2135M:
    def __init__(self, model_path=MODEL_SMOLLM2_135M_PATH):

        # set the device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # set the model
        self.baseModel, self.tokenizer = SmolLm2135M(model_path).setModel()

        self.finetune_name = "SmolLM2-FT-QuantumQA"
        self.finetune_tags = ["smol-course", "module_1"]

    def setSFTConfig(self):

        # Configure the SFTTrainer
        self.sft_config = SFTConfig(
            output_dir=FINE_TUNE_MODEL_PATH,  # Directory to save model checkpoints
            #max_steps=10,  # Adjust based on dataset size and desired training duration
            num_train_epochs=20,  # Adjust based on dataset size and desired training duration
            per_device_train_batch_size=2,  # Set according to your GPU memory capacity
            bf16=True,  # Use bfloat16 for training
            learning_rate=5e-5,  # Common starting point for fine-tuning
            logging_steps=10,  # Frequency of logging training metrics
            save_steps=450,  # Frequency of saving model checkpoints
            evaluation_strategy="steps",  # Evaluate the model at regular intervals
            eval_steps=200,  # Frequency of evaluation
            use_mps_device=(
                True if self.device == "mps" else False
            ),  # Use MPS for mixed precision training
            hub_model_id=self.finetune_name,  # Set a unique name for your model
        )

    def testbasemodel(self):
        #prompt = "Write a haiku about programming"
        prompt = "Can a Turing machine simulate a quantum computer"

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
        self.setSFTConfig()

        # Initialize the SFTTrainer
        trainer = SFTTrainer(
            model=self.baseModel,
            args=self.sft_config,
            train_dataset=train_ds,
            tokenizer=self.tokenizer,
            eval_dataset=test_ds,
            callbacks=[ClearCacheCallback()],
        )
        # print the gpu utilization
        print_gpu_utilization()

        # Train the model
        trainer.train()

        # Save the model
        trainer.save_model(f"./{self.finetune_name}")

    def testfinetunedmodel(self):
        # prompt = "Write a haiku about programming"
        prompt = "Can a Turing machine simulate a quantum computer"

        # Format with template
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # Generate response
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        outputs = self.baseModel.generate(**inputs, max_new_tokens=100)
        print("After training:")
        print(self.tokenizer.decode(outputs[0], skip_special_tokens=True))






