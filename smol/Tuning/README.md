### Reference 
https://github.com/huggingface/smol-course/blob/main/1_instruction_tuning/chat_templates.md

#### Chat templates 
1. Chat templates are essential for structuring interactions between language models and users. 
2. They provide a consistent format for conversations, ensuring that models understand the context and 
3. role of each message while maintaining appropriate response patterns.
4. define how conversations should be formatted when communicating with a language model. 
5. This structure helps maintain consistency across interactions and ensures the model responds 
6. appropriately to different types of inputs. 
7. Below is an example of a chat template:
```
<|im_start|>user
Hi there!<|im_end|>
<|im_start|>assistant
Nice to meet you!<|im_end|>
<|im_start|>user
Can I ask a question?<|im_end|>
<|im_start|>assistant
```

#### Instruct model
1. Instruct model is fine-tuned specifically to follow instructions and engage in conversations.
2. For example, SmolLM2-135M-Instruct is its instruction-tuned variant.
3. when we're using an instruct model we need to make sure we're using the correct chat template.

#### Base model
1. A base model is trained on raw text data to predict the next token. 
2.  For example, SmolLM2-135M
3. To make a base model behave like an instruct model, 
4. we need to format our prompts in a consistent way that the model can understand. 
5. This is where chat templates come in. 
6. ChatML is one such template format that structures conversations 
7. with clear role indicators (system, user, assistant).
8. It's important to note that a base model could be fine-tuned on different chat templates to create instruct models

The tokenize=False parameter in apply_chat_template is used first because 
we want to get the formatted string output first, rather than immediately converting it to token IDs. 
Here's why:

When tokenize=False:
1. The chat template is applied to format the messages into the correct string format
2. Returns a plain text string with all the special tokens and formatting
3. Allows you to see and verify the formatted template before tokenization
4. The actual tokenization happens in the separate tokenizer call afterward


When tokenize=True:
1. It would immediately convert the formatted template into token IDs
2. We wouldn't be able to inspect the intermediate formatted text
3. We'd lose flexibility in how we want to handle the subsequent tokenization



