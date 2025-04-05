import os
# disable Weights and Biases
os.environ['WANDB_DISABLED']="true"


from peft import PeftConfig, PeftModel
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    GenerationConfig
)
# from tqdm import tqdm
# from trl import SFTTrainer
import torch
import time
import pandas as pd
import numpy as np
from huggingface_hub import interpreter_login

interpreter_login()

model_name = "microsoft/phi-2"

eval_tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True, trust_remote_code=True, use_fast=False)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

def gen(model,p, maxlen=100, sample=True):
    toks = eval_tokenizer(p, return_tensors="pt")
    res = model.generate(**toks.to("cuda"), max_new_tokens=maxlen, do_sample=sample,num_return_sequences=1,temperature=0.1,num_beams=1,top_p=0.95,).to('cpu')
    return eval_tokenizer.batch_decode(res,skip_special_tokens=True)

test_ds = load_dataset("mjphayes/machine_learning_questions", split = "test")

compute_dtype = getattr(torch, "float16")
bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )


original_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      device_map='auto',
                                                      quantization_config=bnb_config,
                                                      trust_remote_code=True,
                                                      use_auth_token=True)



checkpoint = "Hunter700/phi2-ml-qa-qlora"
model = PeftModel.from_pretrained(original_model, checkpoint)




questions = test_ds[0:10]['question']
answers = test_ds[0:10]['answer']

original_model_answers = []
peft_model_answers = []

for idx, question in enumerate(questions):

    prompt = f"### Question: {question}\n ### Answer:"

    original_model_res = gen(original_model,prompt,100,)
    original_model_text_output = original_model_res[0].split("### Answer:")[1]

    peft_model_res = gen(model,prompt,100,)
    peft_model_text_output = peft_model_res[0].split('### Answer:')[1]




    original_model_answers.append(original_model_text_output)
    peft_model_answers.append(peft_model_text_output)

zipped_answers = list(zip(answers, original_model_answers, peft_model_answers))

df = pd.DataFrame(zipped_answers, columns = ['Real Answers', 'original_model_answers', 'peft_model_answers'])
df
