from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Function to load the reader model and tokenizer
def load_reader_model(reader_model_name: str) -> pipeline:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        reader_model_name, 
        quantization_config=bnb_config,
        device_map='auto',
        trust_remote_code=True,)
    
    tokenizer = AutoTokenizer.from_pretrained(reader_model_name)

    llm = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )
    return llm

# Function to create a prompt template
def create_prompt_template(tokenizer):
    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context, give a comprehensive answer to the question.
Respond only to the question asked, response should be concise and relevant to the question.
Provide the number of the source document when relevant.
If the answer cannot be deduced from the context, do not give an answer.
"""
        },
        {
            "role": "user",
            "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}

**give in vietnamese language**
"""
        }
    ]
    return tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True)
